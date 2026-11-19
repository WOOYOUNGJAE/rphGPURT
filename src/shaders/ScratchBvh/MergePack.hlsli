#ifndef _MERGE_PACK_HLSLI
#define _MERGE_PACK_HLSLI

// (SinglePass) Leaf-only box merge decisions of CollapseAndEncodeScratchBvh (WideNodeMergingFlags 0x1).
// The same decisions as the 2-pass merge in MergeScratchNodes.hlsli (MergeLeafOnlyBoxNodes, PackLeafOnlyMergeGroups),
// but on a packed dword of box child sizes instead of ScratchWideNodes, so they need no LDS and no ScratchWideNode.
// The min-node packing LUT is LutBuffer.mergePackLut (data in shared/gpurtMergePackLut.h).

#include "../../shadersClean/common/LaneGroup.hlsli"
#include "../../shadersClean/build/BuildCommonScratch.hlsli"

//=====================================================================================================================
// (SinglePass) The functions below take packedSizes, a dword with the box child size of each slot in 4 bits.
//   Size of slot k = (packedSizes >> 4k) & 0xF. Size = numBoxes + numLeaves = min(wide leaf count, 8)
//   Box children occupy slots [0, numBoxSlots). The value is the same across the lane group, so decisions are plain
//   ALU work with no exchange between lanes.
//   SinglePass (CollapseAndEncodeScratchBvh) does not move lanes into slot order, so it uses these instead of the
//   lane based functions of MergeScratchNodes.hlsli.
uint GetPackedSlotSize(
    in uint packedSizes,
    in uint slot)
{
    return (packedSizes >> (4 * slot)) & 0xFu;
}

//=====================================================================================================================
// (SinglePass, WideNodeMergingFlags 0x10) Same decision as PackLeafOnlyMergeGroups without moving lanes.
//   childSlot: Slot of this lane's child. With box children all 8 lanes are filled, so each lane has one of slots 0..7.
//   newSlot: Slot this lane's child moves to. childSlot if unchanged. Non-box children stay in place.
// Returns the mask of slots that start a group, or 0 to keep the original consecutive greedy order.
static uint ComputeLeafOnlyMergePackSlots(
    LaneGroup laneGroup,
    in uint   packedSizes,
    in uint   numBoxSlots,
    in uint   childSlot,
    out uint  newSlot)
{
    newSlot = childSlot;

    GPU_ASSERT(laneGroup.BitOr(1u << childSlot) == bits(8));

    // Non-box slots count as a virtual size 8
    uint childSizes = 0; // childSize of slot k, 4 bits each
    uint numBadBoxes   = 0;
    uint numSmallBoxes = 0; // Slots that can merge with another box
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        const uint sizeK = GetPackedSlotSize(packedSizes, k);
        const bool isBoxK = (k < numBoxSlots);
        numBadBoxes   += (isBoxK && ((sizeK == 0) || (sizeK > 8))) ? 1u : 0u;
        numSmallBoxes += (isBoxK && (sizeK < 8)) ? 1u : 0u;
        childSizes    |= (isBoxK ? sizeK : 8u) << (4 * k);
    }
    if ((numBadBoxes != 0) || (numSmallBoxes < 2))
    {
        return 0; // Fewer than 2 boxes can merge
    }

    // 0. Node count of the original order (consecutive greedy, capacity check only)
    uint baselineCount = 0;
    uint groupSize     = 9; // 9 makes the first box start a new group
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        const uint sizeK = GetPackedSlotSize(childSizes, k);
        if (k < numBoxSlots)
        {
            if ((groupSize + sizeK) > 8)
            {
                baselineCount++;
                groupSize = sizeK;
            }
            else
            {
                groupSize += sizeK;
            }
        }
    }

    // 1. Sort with key = (size - 1, slot). Sorted position = number of smaller keys
    const uint childSize = GetPackedSlotSize(childSizes, childSlot);
    const uint sortKey   = ((childSize - 1) << 3) | childSlot;
    uint sortedPos = 0;
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        const uint keyK = ((GetPackedSlotSize(childSizes, k) - 1) << 3) | k;
        sortedPos += (keyK < sortKey) ? 1u : 0u;
    }

    // 2. Multiset rank = sum_p C(sv_p + p, p + 1), where sv_p = (p-th smallest size) - 1
    const uint sv = childSize - 1;
    uint term = 1;
    [unroll]
    for (uint i = 1; i <= 8; ++i)
    {
        term = (i <= (sortedPos + 1)) ? ((term * (sv + i - 1)) / i) : term;
    }
    const uint rank     = laneGroup.Sum(term); // 0 to 6434. Each lane has one slot, so this is the sum over all slots
    const uint lutEntry = LutBuffer.mergePackLut[rank >> 2][rank & 3];

    // 3. LUT slot that output slot p reads
    // LUT slot order: [groups of sizes 1..7][8s]
    // Output order  : [real 8s (box)][groups of sizes 1..7][virtual 8s (non-box slots, in place)]
    uint numEights = 0; // Real 8s + virtual 8s
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        numEights += (GetPackedSlotSize(childSizes, k) == 8) ? 1u : 0u;
    }
    const uint numSmall      = 8 - numEights;                // LUT slots 0..numSmall-1 hold the groups of sizes 1..7
    const uint numRealEights = numEights - (8 - numBoxSlots);

    uint groupStartMask = 0;
    uint srcSortedPosOfSlots = 0; // Sorted position of the node slot p receives, 3 bits each
    [unroll]
    for (uint p = 0; p < 8; ++p)
    {
        const uint lutSlot = (p < numRealEights) ? (numSmall + p) :
                             ((p < numBoxSlots) ? (p - numRealEights) : p);
        srcSortedPosOfSlots |= ((lutEntry >> (3 * lutSlot)) & 7) << (3 * p);
        if ((p < numBoxSlots) && (((lutEntry >> (24 + lutSlot)) & 1u) != 0))
        {
            groupStartMask |= (1u << p);
        }
    }

    if (countbits(groupStartMask) >= baselineCount)
    {
        return 0; // Already minimal, same as the original merge
    }

    // 4. This lane's child moves to the slot p with its sorted position (positions are unique, so only one matches)
    [unroll]
    for (uint p = 0; p < 8; ++p)
    {
        if (((srcSortedPosOfSlots >> (3 * p)) & 7) == sortedPos)
        {
            newSlot = p;
        }
    }
    GPU_ASSERT((childSlot < numBoxSlots) ? (newSlot < numBoxSlots) : (newSlot == childSlot)); // Non-box slots stay in place

    return groupStartMask;
}

//=====================================================================================================================
// (SinglePass) Leaf-only box merge: the consecutive greedy of MergeLeafOnlyBoxNodes on packedSizes.
// Only capacity (mergedCount > 8) and forcedGroupStartMask decide.
// Returns the mask of follower slots (merged into a preceding sibling).
// Leaders are the box slots that are not followers.
static uint ComputeMergeFollowerMask(
    in uint packedSizes,
    in uint numBoxSlots,
    in uint forcedGroupStartMask)
{
    uint followerSlotMask = 0;
    uint mergedCount    = 0;
    [unroll]
    for (uint k = 0; k < 8; ++k)
    {
        if (k < numBoxSlots)
        {
            const uint sizeK = GetPackedSlotSize(packedSizes, k);
            mergedCount += sizeK;

            // The first box, a preset group boundary or exceeded capacity starts a new group
            const bool shouldCompleteMerge =
                (k == 0) || (((forcedGroupStartMask >> k) & 1u) != 0) || (mergedCount > 8);
            if (shouldCompleteMerge)
            {
                mergedCount = sizeK;
            }
            else
            {
                followerSlotMask |= (1u << k);
            }
        }
    }

    return followerSlotMask;
}

#endif // _MERGE_PACK_HLSLI
