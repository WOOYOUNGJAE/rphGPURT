#ifndef _MERGE_PACK_HLSLI
#define _MERGE_PACK_HLSLI

// (SinglePass) Leaf-only box merge decisions of CollapseAndEncodeScratchBvh (WideNodeMergingFlags 0x1).
// The same decisions as the 2-pass merge in MergeScratchNodes.hlsli (MergeLeafOnlyBoxNodes), but on a packed dword
// of box child sizes instead of ScratchWideNodes, so they need no LDS and no ScratchWideNode.

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
// (SinglePass) Leaf-only box merge: the consecutive greedy of MergeLeafOnlyBoxNodes on packedSizes.
// Only capacity (mergedCount > 8) decides.
// Returns the mask of follower slots (merged into a preceding sibling).
// Leaders are the box slots that are not followers.
static uint ComputeMergeFollowerMask(
    in uint packedSizes,
    in uint numBoxSlots)
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

            // The first box or exceeded capacity starts a new group
            const bool shouldCompleteMerge = (k == 0) || (mergedCount > 8);
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
