#ifndef _MERGE_PACK_HLSLI
#define _MERGE_PACK_HLSLI

// (collapse-time merge) Leaf-only box merge decisions of CollapseAndEncodeScratchBvh, on a packed dword of box child
// sizes (same decisions as MergeLeafOnlyBoxNodes in MergeScratchNodes.hlsli).

#include "../../shadersClean/common/LaneGroup.hlsli"
#include "../../shadersClean/build/BuildCommonScratch.hlsli"

//=====================================================================================================================
// packedSizes: 4 bits per slot, size = min(leaf slot count, 8). Box children use slots [0, numBoxSlots).
uint GetPackedSlotSize(
    in uint packedSizes,
    in uint slot)
{
    return (packedSizes >> (4 * slot)) & 0xFu;
}

//=====================================================================================================================
// Consecutive greedy of MergeLeafOnlyBoxNodes. Returns the follower slot mask (slots merged into the preceding leader).
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
