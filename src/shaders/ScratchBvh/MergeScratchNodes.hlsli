#ifndef _MERGE_SCRATCH_NODES_HLSL
#define _MERGE_SCRATCH_NODES_HLSL


uint GetSharedChildren(uint laneGroupIndex, uint childIndex);
void SetSharedChildren(uint laneGroupIndex, uint childIndex, uint value);

#include "../../shadersClean/common/LaneGroup.hlsli"
#include "../../shadersClean/common/ScratchWideNode.hlsli"
#include "../../shadersClean/common/ObbCommon.hlsli"
#include "../../shadersClean/build/BuildCommonScratch.hlsli"
#include "../EncodeHwBvhCommon3_1.hlsl"

//=====================================================================================================================
static bool ShouldCompleteMerge(
    in uint mergedCount)
{
    if (mergedCount > 8)
    {
        return true;
    }

    return false;
}

//=====================================================================================================================
// Consecutive greedy merging method from "Memory-Efficient BVH" (HPG 2026)
// Fast top-down greedy collapsing guarantees that only leaf-only boxes are merged.
// Returns true on a follower lane (merged into its leader, a preceding sibling).
static bool MergeLeafOnlyBoxNodes(
    LaneGroup                laneGroup,
    in    uint               laneGroupIndex,
    in    uint               scratchWideNodesScratchOffset,
    in    uint               childWideScratchIndex,
    in    bool               childIsBox)
{
    bool isFollowerLane = false;

    // Starts with 11111111. If result of mask is 10010001,
    // that means 0,4,7 are leader nodes, 1,2,3 are followers of 0, and 5,6 are followers of 4.
    uint leaderNodeMask = bits(8); // 0b11111111
    uint childBoxMask = laneGroup.Ballot(childIsBox);
    ScratchWideNode childNode;
    if (childIsBox)
    {
        childNode = FetchScratchWideNode(scratchWideNodesScratchOffset, childWideScratchIndex);
    }
    else
    {
        childNode.children0WideNodeIndexOrLeafScratchIndex = 0xffffffff.xxxx;
        childNode.children1WideNodeIndexOrLeafScratchIndex = 0xffffffff.xxxx;
        childNode.bbox_min = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        childNode.bbox_max = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    
    const uint numBoxesInChild  = childIsBox ? bitFieldExtract(childNode.packedFlags, 0, 4) : 0;
    const uint numLeavesInChild = childIsBox ? bitFieldExtract(childNode.packedFlags, 4, 4) : 0;
    const uint numChildrenInChild = childIsBox ? bitFieldExtract(childNode.packedFlags, 4, 4) : 0;

    // Merge Scratch Wide Node
    uint targetLane = firstbitlow(childBoxMask);
    uint numBoxesInTarget  = laneGroup.Broadcast(numBoxesInChild, targetLane);
    uint numLeavesInTarget = laneGroup.Broadcast(numLeavesInChild, targetLane);
    uint mergedCount = numBoxesInTarget + numLeavesInTarget;
    uint remaining = childBoxMask & ~(1u << targetLane);

    uint numFollowersOfCurLeader = 0; // for debugging
    // Lanegroup broadcasting is safe because all lanes are active
    while (remaining)
    {
        const uint sourceLane = firstbitlow(remaining);
        GPU_ASSERT(targetLane < 8 && sourceLane < 8);

        uint numBoxesInSource  = laneGroup.Broadcast(numBoxesInChild, sourceLane);
        uint numLeavesInSource = laneGroup.Broadcast(numLeavesInChild, sourceLane);
        mergedCount += (numBoxesInSource + numLeavesInSource);

        // Try Merge
        const bool shouldCompleteMerge = ShouldCompleteMerge(mergedCount);
        if (shouldCompleteMerge)
        {
            // Complete current target's merging. re-target to next lane
            targetLane = sourceLane;
            numBoxesInTarget = numBoxesInSource;
            numLeavesInTarget = numLeavesInSource;
            mergedCount = numBoxesInTarget + numLeavesInTarget;
#if SOGANG_EXPERIMENT
            if (numFollowersOfCurLeader > 0)
            {
                const uint sgDebugCountersOffset = ShaderConstants.offsets.sgDebugCounters;
                // ScratchGlobal.InterlockedAdd(sgDebugCountersOffset + numFollowersOfCurLeader * sizeof(uint), 1);
            }
            numFollowersOfCurLeader = 0;
#endif
        }
        else
        {
            // num boxes in child should be 0
            GPU_ASSERT(numBoxesInTarget == 0 && numBoxesInSource == 0);
            // Merge success
            GPU_ASSERT(laneGroup.Ballot(true) == bits(8)); // Ensure all lanes are active and executing the same branch.

            // Update bounding box
            float3 bbox_min = min(laneGroup.Broadcast(childNode.bbox_min, targetLane), laneGroup.Broadcast(childNode.bbox_min, sourceLane));
            float3 bbox_max = max(laneGroup.Broadcast(childNode.bbox_max, targetLane), laneGroup.Broadcast(childNode.bbox_max, sourceLane));

            if (laneGroup.laneIndex == sourceLane)
            {
                [unroll]
                for (uint i = 0; i < numLeavesInSource; ++i)
                {
                    if (i < 4)
                    {
                        SetSharedChildren(laneGroupIndex, i, childNode.children0WideNodeIndexOrLeafScratchIndex[i]);
                    }
                    else
                    {
                        SetSharedChildren(laneGroupIndex, i, childNode.children1WideNodeIndexOrLeafScratchIndex[i - 4]);
                    }
                }

                childNode.packedFlags = bitFieldInsert(childNode.packedFlags, 4, 4, 0);
                childNode.packedFlags = bitFieldInsert(childNode.packedFlags, 8, 1, 0);
                childNode.packedFlags = bitFieldInsert(childNode.packedFlags, 17, 3, numLeavesInTarget);
                isFollowerLane = true;
            }
            GroupMemoryBarrier();
            if (laneGroup.laneIndex == targetLane)
            {
                for (uint i = numLeavesInTarget; i < mergedCount; ++i)
                {
                    const uint sharedOffset = i - numLeavesInTarget;
                    if (i < 4)
                    {
                        childNode.children0WideNodeIndexOrLeafScratchIndex[i] = GetSharedChildren(laneGroupIndex, sharedOffset);
                    }
                    else
                    {
                        childNode.children1WideNodeIndexOrLeafScratchIndex[i - 4] = GetSharedChildren(laneGroupIndex, sharedOffset);
                    }
                }

                // bounding box
                childNode.bbox_min = bbox_min;
                childNode.bbox_max = bbox_max;

                // child count
                childNode.packedFlags = bitFieldInsert(childNode.packedFlags, 4, 4, numLeavesInTarget + numLeavesInSource);
                const uint numFollowersOfTarget = bitFieldExtract(childNode.packedFlags, 17, 3);
                childNode.packedFlags = bitFieldInsert(childNode.packedFlags, 17, 3, numFollowersOfTarget + 1);
#if SOGANG_EXPERIMENT
                numFollowersOfCurLeader++;
#endif
            }
            numLeavesInTarget += numLeavesInSource;
            // Mark source node as a merge candidate. (sourceLane -> 0)
            leaderNodeMask &= ~(1u << sourceLane);

            // Clear SharedChildren
            SetSharedChildren(laneGroupIndex, laneGroup.laneIndex, 0xffffffff);
            GroupMemoryBarrier();
        }
        remaining &= ~(1u << sourceLane);
    }
#if SOGANG_EXPERIMENT
    // Flush the last leader group's count, which the loop above never flushes
    // because it only flushes on overflow (re-target), not on normal loop exit.
    if (numFollowersOfCurLeader > 0)
    {
        const uint sgDebugCountersOffset = ShaderConstants.offsets.sgDebugCounters;
        // // ScratchGlobal.InterlockedAdd(sgDebugCountersOffset + numFollowersOfCurLeader * sizeof(uint), 1);
    }
#endif

    // Clear the leaf bit in the mask.
    leaderNodeMask &= childBoxMask;

    // Update leader lane's node info
    // Mark the current lane as an leader node.
    if (childIsBox)
    {
        const bool isLeaderLane = (leaderNodeMask & (1u << laneGroup.laneIndex)) != 0;
        if (isLeaderLane)
        {
            WriteScratchWideNode(scratchWideNodesScratchOffset, childWideScratchIndex, childNode);
        }
        else
        {
            WriteScratchWideNodeData(scratchWideNodesScratchOffset, childWideScratchIndex, SCRATCH_WIDE_NODE_FLAGS_OFFSET, childNode.packedFlags);
        }
    }
    return isFollowerLane;
}

#endif // _MERGE_SCRATCH_NODES_HLSL