/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
#ifndef _ENCODE_SCRATCH_BVH_HLSL
#define _ENCODE_SCRATCH_BVH_HLSL
#define USE_LDS_CHILDREN 1 // this is default setting
#define FOLLOWER_STACK_ENTRY 0xFFFFFFFE
//=====================================================================================================================
#include "../../shadersClean/common/ShaderDefs.hlsli"

#define GC_DSTBUFFER
#define GC_DSTMETADATA
#define GC_SCRATCHBUFFER
#include "../../shadersClean/build/BuildRootSignature.hlsli"

#define TASK_COUNTER_BUFFER   ScratchGlobal
#define TASK_COUNTER_OFFSET   (ShaderConstants.offsets.taskLoopCounters + TASK_LOOP_WIDE_BVH_COUNTER_OFFSET)
#define NUM_TASKS_DONE_OFFSET (ShaderConstants.offsets.taskLoopCounters + TASK_LOOP_WIDE_BVH_TASKS_DONE_OFFSET)
#include "../TaskMacros.hlsl"

template<typename T>
T LoadInstanceDescBuffer(uint offset)
{
    return InstanceDescBuffer.Load<T>(offset);
}
#include "../IndirectArgBufferUtils.hlsl"

#define LANE_GROUP_SIZE 8
#define MAX_ELEMENTS_PER_THREAD 8
#define MAX_LDS_ELEMENTS_PER_THREADGROUP (MAX_ELEMENTS_PER_THREAD * BUILD_THREADGROUP_SIZE)
groupshared uint SharedMem[MAX_LDS_ELEMENTS_PER_THREADGROUP];
uint GetSharedMem(uint index)
{
    return SharedMem[index];
}
void SetSharedMem(uint index, uint value)
{
    SharedMem[index] = value;
}

uint GetSharedChildren(uint laneGroupIndex, uint childIndex)
{
    return SharedMem[laneGroupIndex * LANE_GROUP_SIZE + childIndex];
}
void SetSharedChildren(uint laneGroupIndex, uint childIndex, uint value)
{
    SharedMem[laneGroupIndex * LANE_GROUP_SIZE + childIndex] = value;
}

#include "../../shadersClean/build/BuildCommonScratch.hlsli"

#if GPURT_BUILD_RTIP3_1

#include "../EncodeHwBvhCommon3_1.hlsl"
#include "../../shadersClean/common/ScratchWideNode.hlsli"
#include "MergeScratchNodes.hlsli"

namespace Gfx12
{

//=====================================================================================================================
// Returns true to enable a separate leaf compression pass over all children of one internal node.
bool EnableLeafCompressionPass()
{
    return (IsTrianglePrimitiveBuild() || IsCompressedTrianglePrimitiveBuild()) &&
           ((Settings.primCompressionFlags & PrimCompFlags::MultiPrim) ||
            (Settings.maxPrimRangeSize > 2));
}

//=====================================================================================================================
// Returns true to enable the optimized wave parallel compression pass (only supports wave32 and maxPrimRangeSize<=4)
bool EnableWaveLeafCompressionPass()
{
#if BUILD_PARALLEL
    return false;
#else
    return EnableLeafCompressionPass();
#endif
}

//=====================================================================================================================
// Pack destination node index (128-byte chunk index) and child index in parent (maximum value of 7) into a single
// DWORD in global stack
uint PackStackDstNodeIdx(
    uint dstNodeIdx,
    uint indexInParent)
{
    return (dstNodeIdx << 3) | indexInParent;
}

//=====================================================================================================================
uint GetStackDstIdx(
    uint packedEntry)
{
    return (packedEntry >> 3);
}

//=====================================================================================================================
uint GetStackDstIdxInParent(
    uint packedEntry)
{
    return (packedEntry & bits(3));
}

//=====================================================================================================================
static void WritePackedExpDisableBoxSortChildIdxAndCount(
    in uint  nodeOffset,
    in uint3 exponents,
    in uint  disableBoxSort,
    in uint  indexInParent,
    in uint  validChildCount)
{
    const uint packedData =
        QuantizedBVH8BoxNode::PackExpChildIdxAndCount(exponents, disableBoxSort, indexInParent, validChildCount);

    DstBuffer.Store(nodeOffset + QUANTIZED_BVH8_NODE_OFFSET_EXP_CHILD_IDX_AND_VALID_COUNT, packedData);
}


//=====================================================================================================================
static uint3 ComputePackedChildInfo(
    in BoundingBox   childBounds,
    in uint          childScratchNodePackedFlags,
    in uint          childScratchNodeIdx,
    in uint          nodeType,
    in uint          nodeRangeLength,
    in float3        rcpExponents,
    in float3        origin,
    in uint          numPrimRefs)
{
    const uint instanceMask =
        ExtractScratchNodeInstanceMask(childScratchNodePackedFlags);

    const uint boxNodeFlags =
        ExtractScratchNodeBoxFlags(childScratchNodePackedFlags);

    const uint3 childInfo = QBVH8::ComputePackedChildInfo(childBounds,
                                                          instanceMask,
                                                          boxNodeFlags,
                                                          nodeType,
                                                          nodeRangeLength,
                                                          rcpExponents,
                                                          origin);
    return childInfo;
}

//=====================================================================================================================
static void EncodeCompressedBoxNode(
    LaneGroup                laneGroup,
    in    uint               rootNodeIndex,
    in    uint               scratchNodesScratchOffset,
    in    uint               scratchWideNodesScratchOffset,
    in    uint               localId,
    in    uint               parentWideScratchNodeIndex,
    in    uint               bvhNodeDstIdx,
    in    uint               currNodePtr,
    in    uint               currNodeOffset,
    in    uint               indexInParent,
    in    AccelStructOffsets offsets,
    in    uint               maxNumChildren,
    in    uint               numPrimRefs,
    in    bool               isRebraidRootNode)
{
    uint baseDstBoxIdx      = 0; // node's first box child's index from total nodes
    uint numLeafChildren    = 0; // num of leaf of parent node
    uint numBoxChildren     = 0; // num of box of parent node
    uint dstStackIdx        = 0;
    uint baseLeafNodeOffset = 0; // node's first leaf child's byte offset
    uint baseBoxNodeOffset  = 0; // node's first box child's byte offset (allBoxOffset + baseDstBoxIdx * sizeof(QBVH8::Node))
    float3 rcpExponents     = 0;
    uint3 exponents;
    BoundingBox parentBbox;
    const uint laneGroupIndex = localId / LANE_GROUP_SIZE; // lane group index within the thread group, not within the wave.

    uint parentScratchNodeIdx = INVALID_IDX;
    uint childWideScratchIndex;
#if USE_LDS_CHILDREN
    // clear shared memory
    [unroll]
    for (uint i = localId; i < MAX_LDS_ELEMENTS_PER_THREADGROUP; i += BUILD_THREADGROUP_SIZE)
    {
        SetSharedMem(i, INVALID_IDX);
    }
    GroupMemoryBarrier();
#else
// USE Reigster children
    uint children8[8] = { INVALID_IDX, INVALID_IDX, INVALID_IDX, INVALID_IDX, INVALID_IDX, INVALID_IDX, INVALID_IDX, INVALID_IDX };
#endif
    uint parentWideNodePackedFlags;
    // Initialization for the first lane
    if (laneGroup.IsFirstLane())
    {
        const ScratchWideNode scratchWideNode = FetchScratchWideNode(scratchWideNodesScratchOffset, parentWideScratchNodeIndex);
        parentScratchNodeIdx = scratchWideNode.scratchIndex;
#if USE_LDS_CHILDREN
        SetSharedChildren(laneGroupIndex, 0, scratchWideNode.children0WideNodeIndexOrLeafScratchIndex.x);
        SetSharedChildren(laneGroupIndex, 1, scratchWideNode.children0WideNodeIndexOrLeafScratchIndex.y);
        SetSharedChildren(laneGroupIndex, 2, scratchWideNode.children0WideNodeIndexOrLeafScratchIndex.z);
        SetSharedChildren(laneGroupIndex, 3, scratchWideNode.children0WideNodeIndexOrLeafScratchIndex.w);

        SetSharedChildren(laneGroupIndex, 4, scratchWideNode.children1WideNodeIndexOrLeafScratchIndex.x);
        SetSharedChildren(laneGroupIndex, 5, scratchWideNode.children1WideNodeIndexOrLeafScratchIndex.y);
        SetSharedChildren(laneGroupIndex, 6, scratchWideNode.children1WideNodeIndexOrLeafScratchIndex.z);
        SetSharedChildren(laneGroupIndex, 7, scratchWideNode.children1WideNodeIndexOrLeafScratchIndex.w);
#else
        children8[0] = scratchWideNode.children0WideNodeIndexOrLeafScratchIndex.x;
        children8[1] = scratchWideNode.children0WideNodeIndexOrLeafScratchIndex.y;
        children8[2] = scratchWideNode.children0WideNodeIndexOrLeafScratchIndex.z;
        children8[3] = scratchWideNode.children0WideNodeIndexOrLeafScratchIndex.w;

        children8[4] = scratchWideNode.children1WideNodeIndexOrLeafScratchIndex.x;
        children8[5] = scratchWideNode.children1WideNodeIndexOrLeafScratchIndex.y;
        children8[6] = scratchWideNode.children1WideNodeIndexOrLeafScratchIndex.z;
        children8[7] = scratchWideNode.children1WideNodeIndexOrLeafScratchIndex.w;
#endif
        parentWideNodePackedFlags = scratchWideNode.packedFlags;
        // GPU_ASSERT(bitFieldExtract(parentWideNodePackedFlags, 8, 1) == 1);
        numBoxChildren = bitFieldExtract(parentWideNodePackedFlags, 0, 4);
        GPU_ASSERT(numBoxChildren <= 8);
    }
#if USE_LDS_CHILDREN
    childWideScratchIndex = GetSharedChildren(laneGroupIndex, laneGroup.laneIndex);
#else
    [unroll]
    for (uint i = 0; i < 8; ++i)
    {
        children8[i] = laneGroup.ReadFirstLane(children8[i]);
    }
    childWideScratchIndex = children8[laneGroup.laneIndex];
#endif
    parentWideNodePackedFlags = laneGroup.ReadFirstLane(parentWideNodePackedFlags);

    parentScratchNodeIdx = laneGroup.ReadFirstLane(parentScratchNodeIdx);
    numBoxChildren = laneGroup.ReadFirstLane(numBoxChildren);
    bool isValidChild = (childWideScratchIndex != INVALID_IDX);
    const bool childIsBox = (laneGroup.laneIndex < numBoxChildren) && (childWideScratchIndex != INVALID_IDX);
    const bool childIsLeaf = (laneGroup.laneIndex >= numBoxChildren) && (childWideScratchIndex != INVALID_IDX);
    numLeafChildren = laneGroup.BallotCount(childIsLeaf);

    uint boxOrLeafChildIndex = childIsBox ? laneGroup.laneIndex : laneGroup.laneIndex - numBoxChildren;

    bool isFollower = MergeLeafOnlyBoxNodes(laneGroup,
                                            laneGroupIndex,
                                            scratchWideNodesScratchOffset,
                                            childWideScratchIndex,
                                            childIsBox);
    const bool isLeaderBoxChild   = childIsBox && !isFollower;
    const bool isFollowerBoxChild = childIsBox && isFollower;

    const uint leaderBoxMask   = laneGroup.Ballot(isLeaderBoxChild);
    const uint followerBoxMask = laneGroup.Ballot(isFollowerBoxChild);

    const uint numLeaderBoxChildren   = countbits(leaderBoxMask);
    const uint numFollowerBoxChildren = countbits(followerBoxMask);

    // Write leader box mask to parent node - not used
    // if (laneGroup.IsFirstLane())
    // {
    //     parentWideNodePackedFlags = bitFieldInsert(parentWideNodePackedFlags, 9, 8, leaderBoxMask & bits(8));
    //     WriteScratchWideNodeData(scratchWideNodesScratchOffset, parentWideScratchNodeIndex, SCRATCH_WIDE_NODE_FLAGS_OFFSET, parentWideNodePackedFlags);
    // }

#if (MERGE_BOX_NODES_FLAG == 1) // disallow compaction
    const uint boxChildIndexInParent = laneGroup.laneIndex;
#elif (MERGE_BOX_NODES_FLAG == 2) // compacted dst bvh
    uint shiftedMask = leaderBoxMask & ((1u << laneGroup.laneIndex) - 1);
    const uint boxChildCompactedIndexInParent = countbits(shiftedMask);
    const uint boxChildIndexInParent = boxChildCompactedIndexInParent;
#endif // (MERGE_BOX_NODES_FLAG == 1)

    if (laneGroup.IsFirstLane())
    {
#if (MERGE_BOX_NODES_FLAG == 1)
        const uint numBoxchildrenToPush = numBoxChildren;
#elif (MERGE_BOX_NODES_FLAG == 2)
        const uint numBoxchildrenToPush = numLeaderBoxChildren;
        // If MERGE_BOX_NODES_FLAG == 1, increment it in the CollapseScratch pass.
        IncrementAccelStructHeaderField(ACCEL_STRUCT_HEADER_NUM_INTERNAL_FP32_NODES_OFFSET, 1);
#endif
        const uint numInlinePrimStructs = 0;
        // This code assumes STACK_PTRS_SRC_PTR_OFFSET and STACK_PTRS_DST_PTR_OFFSET are in consecutive
        // DWORDs.
        const uint stackPtrOffset = ShaderConstants.offsets.qbvhGlobalStackPtrs;
        ScratchGlobal.InterlockedAdd(stackPtrOffset + STACK_PTRS_DST_PTR_OFFSET, numBoxchildrenToPush + numInlinePrimStructs, dstStackIdx);
    }

    if (laneGroup.IsFirstLane())
    {
        baseDstBoxIdx = dstStackIdx;
        baseBoxNodeOffset = offsets.internalNodes + baseDstBoxIdx * QBVH8::NodeSizeInBytes;
    }
    dstStackIdx       = laneGroup.ReadFirstLane(dstStackIdx);
    baseDstBoxIdx     = laneGroup.ReadFirstLane(baseDstBoxIdx);
    baseBoxNodeOffset = laneGroup.ReadFirstLane(baseBoxNodeOffset);

    if (isValidChild)
    {
        GPU_ASSERT(childWideScratchIndex != INVALID_IDX);
        uint childStackIdx = dstStackIdx + boxChildIndexInParent;
        if (isLeaderBoxChild)
        {
            // DstIndex and LocalIndex
            uint packedDstNodeIdx = PackStackDstNodeIdx(baseDstBoxIdx + boxChildIndexInParent, boxChildIndexInParent);

            // Regular operation for other threads with isBox set
            ScratchGlobal.Store2(ShaderConstants.offsets.qbvhGlobalStack + (childStackIdx * sizeof(uint2)),
                            uint2(childWideScratchIndex, packedDstNodeIdx));
        }
#if (MERGE_BOX_NODES_FLAG == 1)
        else if (isFollowerBoxChild)
        {
            ScratchGlobal.Store2(ShaderConstants.offsets.qbvhGlobalStack + (childStackIdx * sizeof(uint2)),
                            uint2(FOLLOWER_STACK_ENTRY, INVALID_IDX));
        }
#endif
    }

    // Write Parent Result Node
    uint basePrimStructIdx = 0;
    const uint validChildCount = laneGroup.BallotCount(isValidChild);
    if (laneGroup.IsFirstLane())
    {
        const ScratchNode parentScratchNode = FetchScratchNode(scratchNodesScratchOffset, parentScratchNodeIdx);
        // Fetch bounding box data from the wide node, as merged nodes may have changed
        parentBbox.min = FetchScratchWideNodeData<float3>(scratchWideNodesScratchOffset, parentWideScratchNodeIndex, SCRATCH_WIDE_NODE_BBOX_MIN_OFFSET);
        parentBbox.max = FetchScratchWideNodeData<float3>(scratchWideNodesScratchOffset, parentWideScratchNodeIndex, SCRATCH_WIDE_NODE_BBOX_MAX_OFFSET);

        // The root of the BVH2 treelet contains the combined bounds of the child nodes in the node array. Compute the
        // common exponent from the root node bounds
        exponents = ComputeCommonExponent(parentBbox.min, parentBbox.max, 12);

        // Pre-compute exponent reciprocal
        rcpExponents = ComputeFastExpReciprocal(exponents, 12);

        QBVH8::WriteInternalNodeBaseOffset(currNodeOffset, baseBoxNodeOffset);
        if (EnableWaveLeafCompressionPass() == false)
        {
            QBVH8::WriteLeafNodeBaseOffset(currNodeOffset, baseLeafNodeOffset);
        }
        QBVH8::WriteLeafNodeBaseOffset(currNodeOffset, INVALID_NODE);

        QBVH8::WriteOrigin(currNodeOffset, parentBbox.min);

#if 0
        // The box sort disabled bit indicates that merged-node de-duplication is enabled instead of the default de-duplication method.
        const uint deduplicate = (uint)(numBoxChildren != numLeaderBoxChildren);
        WritePackedExpDisableBoxSortChildIdxAndCount(currNodeOffset, exponents, deduplicate, indexInParent, validChildCount);
#else
        QBVH8::WritePackedExpChildIdxAndCount(currNodeOffset, exponents, indexInParent, validChildCount);
#endif
        QBVH8::WriteObbMatrixIndex(currNodeOffset, INVALID_OBB);
    }

    parentBbox.min    = laneGroup.ReadFirstLane(parentBbox.min);
    parentBbox.max    = laneGroup.ReadFirstLane(parentBbox.max);
    exponents         = laneGroup.ReadFirstLane(exponents);
    rcpExponents      = laneGroup.ReadFirstLane(rcpExponents);
    basePrimStructIdx = laneGroup.ReadFirstLane(basePrimStructIdx);

    ScratchNode childScratchNode;
    uint childScratchNodeIndex = INVALID_IDX;
    if (childIsBox)
    {
        childScratchNodeIndex = FetchScratchWideNodeData<uint>(scratchWideNodesScratchOffset, childWideScratchIndex, SCRATCH_WIDE_NODE_SCRATCH_INDEX_OFFSET);
    }
    else if (childIsLeaf)
    {
        childScratchNodeIndex = childWideScratchIndex;
    }

    if (isValidChild)
    {
        // TLAS builds initialize these fields immediately. They should be left 0 for BLAS builds if we are compressing
        // leaf nodes. If we are not compressing, a single triangle primitive structure is produced immediately.
        // Either way the triangle node type field is 0.
        const uint nodeType        = childIsLeaf ? NODE_TYPE_TRIANGLE_0 : NODE_TYPE_BOX_QUANTIZED_BVH8;
        const uint nodeRangeLength = childIsLeaf ? (EnableLeafCompressionPass() ? 0 : 1) :
                                                    // During the decode stage of traverse, nodeRangeLength indicates how many
                                                    // internal nodes away the next sibling is from this boxNode. If the right
                                                    // sibling is a follower, increment it by 0 so that it points to the
                                                    // same node as the left sibling.
                                                    ((leaderBoxMask & (1u << (laneGroup.laneIndex + 1))) == 0) ? 0 :
#if (MERGE_BOX_NODES_FLAG == 1)
                                                    // (Number of consecutive follower (0) bits ending at laneIndex) + 1.
                                                    (laneGroup.laneIndex + 1 - firstbithigh(leaderBoxMask & bits(laneGroup.laneIndex + 1)));
#elif (MERGE_BOX_NODES_FLAG == 2)
                                                    1; // leader nodes are consecutive
#endif
        const uint baseNodeOffset  = childIsLeaf ? baseLeafNodeOffset : baseBoxNodeOffset;
        const uint offset          = childIsLeaf ? boxOrLeafChildIndex + numBoxChildren : boxOrLeafChildIndex;
        const uint childNodeOffset = baseNodeOffset + 
                (childIsLeaf ? boxOrLeafChildIndex : boxChildIndexInParent) * QBVH8::NodeSizeInBytes;

        BoundingBox childBbox;
        childScratchNode = FetchScratchNode(scratchNodesScratchOffset, childScratchNodeIndex);
        childBbox = GetScratchNodeBoundingBoxTS(scratchNodesScratchOffset, IsLeafNode(childScratchNodeIndex, numPrimRefs), childScratchNode);

        // uint childPackedFlags = FetchScratchNodeData<uint>(scratchNodesScratchOffset, childScratchNodeIndex, SCRATCH_NODE_FLAGS_OFFSET);
        const uint3 childInfo = ComputePackedChildInfo(childBbox,
                                                       childScratchNode.packedFlags,
                                                       childScratchNodeIndex,
                                                       nodeType,
                                                       nodeRangeLength,
                                                       rcpExponents,
                                                       parentBbox.min,
                                                       numPrimRefs);
        QBVH8::WritePackedChildInfo(currNodeOffset, offset, childInfo);

        if (isLeaderBoxChild)
        {
            QBVH8::WriteParentPointer(childNodeOffset, currNodePtr);
        }
    }
    else
    {
        QBVH8::WriteInvalidChildInfo(currNodeOffset, laneGroup.laneIndex);
    }

    if (IsObbEnabled())
    {
        float aabbSa = 0.0f;
        if (childScratchNodeIndex != INVALID_IDX)
        {
            aabbSa = GetScratchNodeSurfaceArea(
                scratchNodesScratchOffset, childScratchNodeIndex, childScratchNode, numPrimRefs);
        }

        const float aabbSum = laneGroup.Sum(aabbSa);
        const float aabbMax = laneGroup.Max(aabbSa);
        const uint aabbMaxChildIdx = laneGroup.GetFirstActiveLaneIndex(aabbSa == aabbMax);

        // (Loosen DOBB, deprecated) Pick the OBB child with each block's surface area split over its members, so a
        // child outside any block can win over a block. Without LOOSEN_DOBB these equal the original values.
        uint  loosenObbChildIdx = aabbMaxChildIdx; // child whose orientation this node uses
        float loosenAabbSum     = aabbSum;         // AABB surface area sum the refit compares the OBB against
#if LOOSEN_DOBB
        if (IsLoosenDobbEnabled())
        {
            uint numBlockMembers = 1;
            if (childIsBox)
            {
                // A block runs from its leader up to the next child that is not a follower. Lane 0 is always a leader.
                const uint notFollowerMask = ~followerBoxMask & bits(LANE_GROUP_SIZE);
                const uint laneAndBelow    = bits(laneGroup.laneIndex + 1);
                const uint blockLeaderLane = firstbithigh(notFollowerMask & laneAndBelow);
                const uint nextLeaderLane  = firstbitlow((notFollowerMask | (1u << LANE_GROUP_SIZE)) & ~laneAndBelow);
                numBlockMembers = nextLeaderLane - blockLeaderLane;
                GPU_ASSERT(numBlockMembers > 0);
            }

            const float loosenSa  = aabbSa / float(numBlockMembers);
            const float loosenMax = laneGroup.Max(loosenSa);
            loosenAabbSum     = laneGroup.Sum(loosenSa);
            loosenObbChildIdx = laneGroup.GetFirstActiveLaneIndex(loosenSa == loosenMax);
        }
#endif

        // (Best4) Merged leaf-only node: every leaf's matrix is a candidate. numBoxChildren and the follower count
        // (packedFlags bits 17..19) are the same on all lanes.
        const bool collectObbCandidates = IsObbBest4Enabled() && (numBoxChildren == 0) &&
                                          (bitFieldExtract(parentWideNodePackedFlags, 17, 3) != 0);
        // (Best4 on all nodes) Every other node up to the root: its 4 largest children, as without merging.
        const bool collectChildObbCandidates = IsObbBest4Enabled() && IsObbBest4AllNodes() &&
                                               (collectObbCandidates == false);

        uint bestObbMatrixIdx = INVALID_OBB;
        if (((aabbMaxChildIdx == laneGroup.laneIndex) || (loosenObbChildIdx == laneGroup.laneIndex) ||
             collectObbCandidates || collectChildObbCandidates) && (childIsLeaf))
        {
            uint packedFlags = childScratchNode.packedFlags;

            bool isCollapse = childScratchNode.numPrimitivesAndDoCollapse & 1;

            if (CollapseAnyPairs())
            {
                isCollapse = (childScratchNode.numPrimitivesAndDoCollapse >> 1) == 2;
            }

            if (isCollapse)
            {
                // Triangles in a range are colocated and tend to share a similar orientation, so
                // choosing one at random from the range is about as effective as choosing one with
                // a more sophisticated heuristic.
                //
                // Pick the first triangle in the range as an approximation.
                uint triNodeIdx = childScratchNode.left_or_primIndex_or_instIndex;

                if (UsePrimIndicesArray())
                {
                    triNodeIdx = GetPrimRefIdx(FetchSortedPrimIndex(ShaderConstants.offsets.primIndicesSorted,
                                                                    childScratchNode.sortedPrimIndex), numPrimRefs);
                }

                packedFlags =
                    FetchScratchNodeData<uint>(scratchNodesScratchOffset, triNodeIdx, SCRATCH_NODE_FLAGS_OFFSET);
            }

            bestObbMatrixIdx = ExtractScratchNodeObbMatrixIdx(packedFlags);
        }

        const uint leafObbMatrixIdx = bestObbMatrixIdx; // this lane's leaf matrix (when computed above)

        // Leaf matrix of the loosened best child (same as the original one when Loosen DOBB is off)
        const uint loosenObbMatrixIdx = laneGroup.Broadcast(bestObbMatrixIdx, loosenObbChildIdx);
        bestObbMatrixIdx = laneGroup.Broadcast(bestObbMatrixIdx, aabbMaxChildIdx);

        // (Best4) Candidates for the refit (OBB_TABLE_CANDIDATES_OFFSET); candidate 0 is the current choice
        uint obbCandidates = OBB_CANDIDATES_ALL_EMPTY;
        if (collectObbCandidates)
        {
            // Merged leaf-only node: the next largest leaves' matrices, without duplicates
            obbCandidates = bitFieldInsert(obbCandidates, 0, 8, loosenObbMatrixIdx);

            bool isUnlisted = childIsLeaf && (leafObbMatrixIdx != INVALID_OBB) &&
                              (leafObbMatrixIdx != loosenObbMatrixIdx);
            [unroll]
            for (uint k = 1; k < OBB_NUM_MATRIX_CANDIDATES; ++k)
            {
                const float unlistedSa  = isUnlisted ? aabbSa : -1.0f;
                const float unlistedMax = laneGroup.Max(unlistedSa);
                if (unlistedMax >= 0.0f) // same on all lanes
                {
                    const uint lane   = laneGroup.GetFirstActiveLaneIndex(isUnlisted && (unlistedSa == unlistedMax));
                    const uint matrix = laneGroup.Broadcast(leafObbMatrixIdx, lane);
                    obbCandidates = bitFieldInsert(obbCandidates, 8 * k, 8, matrix);
                    isUnlisted    = isUnlisted && (leafObbMatrixIdx != matrix);
                }
            }
        }
        else if (collectChildObbCandidates)
        {
            // As in EncodeHwBvh3_1, with a box child named by its lane (= its refit lane).
            const uint laneSlot = childIsLeaf ? leafObbMatrixIdx : (OBB_CANDIDATE_BOX_LANE_BIT | laneGroup.laneIndex);
            obbCandidates = bitFieldInsert(obbCandidates, 0, 8, laneGroup.Broadcast(laneSlot, loosenObbChildIdx));

            // The members of a merged block get the same matrix from the refit, so a block takes one candidate.
            // childBlock is the lane of the block leader, and the lane itself for every other child.
            const uint childBlock = firstbithigh(~followerBoxMask & bits(laneGroup.laneIndex + 1));

            bool isUnpicked = (childIsLeaf || childIsBox) &&
                              (childBlock != laneGroup.Broadcast(childBlock, loosenObbChildIdx)) &&
                              (laneSlot != OBB_CANDIDATE_EMPTY);
            [unroll]
            for (uint k = 1; k < OBB_NUM_MATRIX_CANDIDATES; ++k)
            {
                const float unpickedSa  = isUnpicked ? aabbSa : -1.0f;
                const float unpickedMax = laneGroup.Max(unpickedSa);
                if (unpickedMax >= 0.0f) // same on all lanes
                {
                    const uint lane = laneGroup.GetFirstActiveLaneIndex(isUnpicked && (unpickedSa == unpickedMax));
                    obbCandidates = bitFieldInsert(obbCandidates, 8 * k, 8, laneGroup.Broadcast(laneSlot, lane));
                    isUnpicked    = isUnpicked && (childBlock != laneGroup.Broadcast(childBlock, lane));
                }
            }
        }

        if (laneGroup.IsFirstLane())
        {
            uint leafStartMask = 0;

            if (numBoxChildren == 0)
            {
                // Push children based on their original counts before merging, rather than as a single merged entry.

                // followers merged into this (leader) node
                const uint numFollowers = bitFieldExtract(parentWideNodePackedFlags, 17, 3);

                // original children in leader node
                uint obbRefitStackIndex = 0;
                ScratchGlobal.InterlockedAdd(
                    ShaderConstants.offsets.obbRefitStackPtrs + STACK_PTRS_OBB_REFIT_BATCH_COUNT, 1, obbRefitStackIndex);

                bool isBlockNode = numFollowers != 0;

                // The stack entry is a plain node index. How a block is partitioned is described by the leaf start
                // mask in the OBB flags below, so nothing has to be packed in here.
                ScratchGlobal.Store(
                    ShaderConstants.offsets.obbRefitStack + (obbRefitStackIndex * sizeof(uint)), bvhNodeDstIdx);

                if (isBlockNode)
                {
                    // The leader always owns the first segment, so bit 0 is always set.
                    leafStartMask = 1u;

                    for (uint i = 1; i <= numFollowers; ++i)
                    {
                        const uint followerPackedFlags =
                            FetchScratchWideNodeData<uint>(scratchWideNodesScratchOffset, parentWideScratchNodeIndex + i, SCRATCH_WIDE_NODE_FLAGS_OFFSET);

                        const uint segmentBegin = bitFieldExtract(followerPackedFlags, 17, 3);
                        GPU_ASSERT((segmentBegin > 0) && (segmentBegin < 8));
                        GPU_ASSERT((leafStartMask & (1u << segmentBegin)) == 0);

                        leafStartMask |= (1u << segmentBegin);
                    }
                }
            }

            // OBB refit flags are packed as follows:
            // b[0:3]   : 2's complement of the (leader)box child count. The maximum value can be 8. That requires 3
            //           bits plus the sign bit --> 4 bits. The maximum carry bits (for when box count is
            //           incremented by refit kernel) is 1, but we have plenty more room in unused bits beside.
            // b[4:13]  : 2's complement increment overflow bit
            // b[14:21] : Leaf start mask. Bit k is set when a block member's leaf segment starts at leaf k.
            //           Bit 0 is always set for the leader, so countbits() is the block member count and the
            //           distance to the next set bit is a member's leaf count. 0 means this is not a block node.
            // b[22:24] : Child index with maximum AABB surface area for this node.
            // b[25:31] : OBB matrix index of the best leaf node amongst the children. This is precomputed and stored in
            //           scratch node during morton code generation phase to absorb the VGPR impact of that computation.
            if (IsObbTableExtended())
            {
                // 16 byte entry: loosened best child and split surface area sum, then the original best child (passed
                // up to the parent with Loosen DOBB) and the candidates. Without Loosen DOBB the loosened values equal
                // the original ones.
                const uint encodedBits = (loosenObbMatrixIdx << OBB_FLAGS_BIT_OFFSET_OBB_MATRIX_INDEX) |
                                         (loosenObbChildIdx  << OBB_FLAGS_BIT_OFFSET_AABB_MAX_CHILD_INDEX) |
                                         (leafStartMask      << OBB_FLAGS_BIT_OFFSET_LEAF_START_MASK) |
                                         ((numLeaderBoxChildren ^ 0xFu) + 1);
                const uint origBest = (bestObbMatrixIdx << 3) | aabbMaxChildIdx;

                const uint obbFlagsOffset =
                    ShaderConstants.offsets.qbvhObbFlags + (bvhNodeDstIdx * OBB_TABLE_EXTENDED_ENTRY_SIZE);
                ScratchGlobal.Store<uint4>(obbFlagsOffset,
                                           uint4(encodedBits, asuint(loosenAabbSum), origBest, obbCandidates));
            }
            else
            {
                const uint encodedBits = (bestObbMatrixIdx << OBB_FLAGS_BIT_OFFSET_OBB_MATRIX_INDEX) |
                                         (aabbMaxChildIdx  << OBB_FLAGS_BIT_OFFSET_AABB_MAX_CHILD_INDEX) |
                                         (leafStartMask    << OBB_FLAGS_BIT_OFFSET_LEAF_START_MASK) |
                                         ((numLeaderBoxChildren ^ 0xFu) + 1);

                const uint obbFlagsOffset =
                    ShaderConstants.offsets.qbvhObbFlags + (bvhNodeDstIdx * OBB_TABLE_ENTRY_SIZE);
                ScratchGlobal.Store<uint2>(obbFlagsOffset, uint2(encodedBits, asuint(aabbSum)));
            }
        }
    }

    if (laneGroup.IsFirstLane())
    {
        ScratchGlobal.InterlockedAdd(ShaderConstants.offsets.wideNodeEncodingDoneCounter, 1 + numFollowerBoxChildren);
    }
    uint localPrimCount = 0;
    if (childScratchNodeIndex != INVALID_IDX)
    {
        if (childIsLeaf)
        {
            localPrimCount = FetchScratchNodeNumPrimitives(childScratchNode, IsLeafNode(childScratchNodeIndex, numPrimRefs));
        }
    }

    const uint primCount = WaveActiveSum(localPrimCount);

    const uint numEmptySlots = maxNumChildren - (numBoxChildren + numLeafChildren);


    if (EnableWaveLeafCompressionPass() && (numLeafChildren > 0))
    {
        WritePrimCompBatches3_1(laneGroup,
                                rootNodeIndex,
                                scratchNodesScratchOffset,
                                parentScratchNodeIdx,
                                currNodeOffset,
                                numBoxChildren,
                                numLeafChildren,
                                primCount,
                                numPrimRefs,
                                childScratchNodeIndex,
                                boxOrLeafChildIndex,
                                childIsLeaf);
    }

    // Write fat leaf offsets.
    if (IsUpdateAllowed() && (numBoxChildren == 0) && laneGroup.IsFirstLane())
    {
        const uint index = IncrementAccelStructHeaderField(ACCEL_STRUCT_HEADER_NUM_INTERNAL_FP16_NODES_OFFSET, 1);
        const uint baseUpdatePtrOffset = offsets.primNodePtrs + (ShaderConstants.maxNumPrimitives * sizeof(uint));
        DstBuffer.Store(baseUpdatePtrOffset + (index * sizeof(uint)), currNodeOffset);
    }
}

//=====================================================================================================================
// Convert source BVH2 (in scratch memory) into RTIP3.1 compatible hardware format
void EncodeScratchBvhImpl(
    uint globalId,
    uint localId,
    uint numPrimRefs,
    uint numScratchWideNodes)
{
    GPU_ASSERT(EnableWaveLeafCompressionPass());
    const AccelStructOffsets offsets = ShaderConstants.header.offsets;

    const uint rootNodeIndex = GetBvh2RootNodeIndex();
    const uint scratchNodesScratchOffset     = CalculateBvhNodesOffset(ShaderConstants, numPrimRefs);
    const uint scratchWideNodesScratchOffset = ShaderConstants.offsets.bvhWideNodeData;

    const bool isSinglePrimBvh = (numPrimRefs == 1);

    const ScratchNode rootScratchNode = FetchScratchNode(scratchNodesScratchOffset, rootNodeIndex);
    const uint numActivePrims = FetchScratchNodeNumPrimitives(rootScratchNode, isSinglePrimBvh);

    if (IsObbEnabled())
    {
        if (globalId < numScratchWideNodes)
        {
            if (IsObbTableExtended())
            {
                ScratchGlobal.Store<uint4>(
                    ShaderConstants.offsets.qbvhObbFlags + globalId * OBB_TABLE_EXTENDED_ENTRY_SIZE, 0);
            }
            else
            {
                ScratchGlobal.Store<uint2>(ShaderConstants.offsets.qbvhObbFlags + globalId * OBB_TABLE_ENTRY_SIZE, 0);
            }
        }
    }
    if (globalId == 0)
    {
        const BoundingBox bbox =
            GetScratchNodeBoundingBoxTS(scratchNodesScratchOffset, isSinglePrimBvh, rootScratchNode);

        WriteAccelStructHeaderRootBoundingBox(bbox);

        const uint3 exponents = ComputeCommonExponent(bbox.min, bbox.max, 12);

        // Pre-compute exponent reciprocal
        const float3 rcpExponents = ComputeFastExpReciprocal(exponents, 12);

        // Generate intersectable instance node data in BLAS metadata
        if ((Settings.instanceMode == InstanceMode::Passthrough) || isSinglePrimBvh)
        {
            const uint instanceMask =
                ExtractScratchNodeInstanceMask(rootScratchNode.packedFlags);

            const uint blasRootNodePointer = CreateRootNodePointer3_1();
            WriteInstancePassthrough(blasRootNodePointer,
                                        bbox,
                                        ACCEL_STRUCT_METADATA_INSTANCE_NODE_OFFSET);
        }
        else if (Settings.instanceMode == InstanceMode::FilterNode)
        {
            // Generate filter node data in BLAS metadata
            WriteBlasInstanceFilterNode(scratchNodesScratchOffset,
                                        numPrimRefs,
                                        rootNodeIndex,
                                        bbox.min,
                                        exponents,
                                        rcpExponents);
        }

        DstBuffer.Store(ACCEL_STRUCT_HEADER_PACKED_FLAGS_OFFSET, rootScratchNode.packedFlags);

        // Write parent pointer for root node.
        QBVH8::WriteParentPointer(offsets.internalNodes, INVALID_IDX);
    }

    LaneGroup laneGroup;
    laneGroup.Alloc(8);

    // Each stack item stores data for writes to linear QBVH memory, indexed by stack index
    const uint globalStackIndex = globalId / 8;

    // With early pairing enabled, numPrimRefs represents the active primitive references. However,
    // we need to use the active primitive count to estimate the hardware internal node count.
    bool isGlobalStackEntryDone = (globalStackIndex >= numScratchWideNodes);

    const uint maxNumChildren = 8;
    while (isGlobalStackEntryDone == false)
    {
        uint numWideNodeDone = 0;
        if (WaveIsFirstLane())
        {
            numWideNodeDone = ScratchGlobal.Load(ShaderConstants.offsets.wideNodeEncodingDoneCounter);
        }
        numWideNodeDone = WaveReadLaneFirst(numWideNodeDone);
        if (numWideNodeDone >= numScratchWideNodes)
        {
            break;
        }

        uint2 stackEntry = uint2(0, 0);
        if (laneGroup.IsFirstLane())
        {
            stackEntry = StackPop(globalStackIndex);
        }

        stackEntry = laneGroup.ReadFirstLane(stackEntry);
#if (MERGE_BOX_NODES_FLAG == 1)
        if (stackEntry.x == FOLLOWER_STACK_ENTRY)
        {
            isGlobalStackEntryDone = true;
        } else
#endif
        // If we have a valid node on the stack, process node
        if (stackEntry.x != INVALID_IDX)
        {
            const uint bvhWideNodeSrcIdx = stackEntry.x; // parentWideScratchNodeIndex
            const uint bvhNodeDstIdx = GetStackDstIdx(stackEntry.y);
            const uint indexInParent = GetStackDstIdxInParent(stackEntry.y);
            const uint currNodeAddr = offsets.internalNodes + (bvhNodeDstIdx * QBVH8::NodeSizeInBytes);
            const uint currNodePtr = CreateBoxNodePointer(currNodeAddr, false);
            
            // Pull up the nodes chosen by SAH to be the new children in BVH8
            EncodeCompressedBoxNode(laneGroup,
                                    rootNodeIndex,
                                    scratchNodesScratchOffset,
                                    scratchWideNodesScratchOffset,
                                    localId,
                                    bvhWideNodeSrcIdx,
                                    bvhNodeDstIdx,
                                    currNodePtr,
                                    currNodeAddr, // currNodeOffset
                                    indexInParent,
                                    offsets,
                                    maxNumChildren,
                                    numPrimRefs,
                                    false);

            isGlobalStackEntryDone = true;
        }
    }
}

} // namespace Gfx12

#endif

#if NO_SHADER_ENTRYPOINT == 0
//=====================================================================================================================
// Main Function : EncodeScratchBvh
//=====================================================================================================================
[RootSignature(RootSig)]
[numthreads(BUILD_THREADGROUP_SIZE, 1, 1)]
void EncodeScratchBvh(
    uint globalIdIn : SV_DispatchThreadID,
    uint groupIdIn  : SV_GroupID,
    uint localIdIn  : SV_GroupThreadID)
{
#if GPURT_BUILD_RTIP3_1 && MERGE_BOX_NODES_FLAG
    uint globalId = globalIdIn;
    uint localId  = localIdIn;
    uint groupId  = groupIdIn;

    const uint numScratchWideNodes = ScratchGlobal.Load(ShaderConstants.offsets.scratchWideNodeCounter);
    const uint numActivePrims = ReadAccelStructHeaderField(ACCEL_STRUCT_HEADER_NUM_ACTIVE_PRIMS_OFFSET);

    uint numTasksWait = 0;
    uint waveId       = 0;

    INIT_TASK;

    BEGIN_TASK(1)
    if (globalId == 0)
    {
        const uint rootWideNodeIndex = 0;
        InitStackPtrs(rootWideNodeIndex);
        Gfx12::InitObbRefitStackPtrs();
    }
    END_TASK(1);
    BEGIN_TASK(RoundUpQuotient(numScratchWideNodes, BUILD_THREADGROUP_SIZE));
    ClearStackPtrs(globalId, numScratchWideNodes, ShaderRootConstants.NumThreads());
    END_TASK(RoundUpQuotient(numScratchWideNodes, BUILD_THREADGROUP_SIZE));

    const uint numThreads = numScratchWideNodes * 8;
    BEGIN_TASK(RoundUpQuotient(numThreads, BUILD_THREADGROUP_SIZE));
    Gfx12::EncodeScratchBvhImpl(globalId, localId, numActivePrims, numScratchWideNodes);
    END_TASK(RoundUpQuotient(numThreads, BUILD_THREADGROUP_SIZE));

#if GPURT_ENABLE_GPU_DEBUG
    BEGIN_TASK(1);
    if (globalId == 0)
    {
        const uint numBoxNodes = DstBuffer.Load(ACCEL_STRUCT_HEADER_NUM_INTERNAL_FP32_NODES_OFFSET);
        GPU_DPF("numOriginalBoxNodes: %u, numDstBoxNodes: %u",numScratchWideNodes, numBoxNodes);
    }
    END_TASK(1);
#endif

    BEGIN_TASK(1);
    PostHwBvhBuild(localId, numActivePrims);

    if (globalId == 0)
    {
        if (Gfx12::EnableWaveLeafCompressionPass())
        {
            // The CompressPrims dispatch requires one group per compression batch plus enough groups to process the
            // entire single prims list
            const uint compressBatches =
                RoundUpQuotient(FetchCompressPrimBatchCount(ShaderConstants.offsets.qbvhGlobalStackPtrs),
                                COMPRESSION_BATCHES_PER_GROUP);

            const uint singlePrimCount  = FetchCompressPrimSingleCount(ShaderConstants.offsets.qbvhGlobalStackPtrs);
            const uint singlePrimGroups = RoundUpQuotient(RoundUpQuotient(singlePrimCount, 2), PRIM_COMP_GROUP_SIZE);

            ScratchGlobal.Store(ShaderConstants.offsets.qbvhGlobalStackPtrs + STACK_PTRS_PRIM_COMP_GROUP_COUNT_X,
                                compressBatches + singlePrimGroups);
        }

        if (IsObbEnabled())
        {
            const uint obbRefitBatchCount =
                ScratchGlobal.Load(ShaderConstants.offsets.obbRefitStackPtrs + STACK_PTRS_OBB_REFIT_BATCH_COUNT);

            // Note, a group of 8 lanes works on a single internal node.
            const uint obbRefitGroupCount = RoundUpQuotient(obbRefitBatchCount,
                                                            REFIT_ORIENTED_BOUNDS3_1_THREADGROUP_SIZE / 8u);
            ScratchGlobal.Store(ShaderConstants.offsets.obbRefitStackPtrs + STACK_PTRS_OBB_REFIT_GROUP_COUNT_X,
                                obbRefitGroupCount);
#if GPURT_ENABLE_GPU_DEBUG
            GPU_DPF("numObbRefitBatches: %u, numObbRefitGroups: %u", obbRefitBatchCount, obbRefitGroupCount);
#endif
        }
    }

#if GPURT_ENABLE_GPU_DEBUG
    END_TASK(1);
#else
    END_TASK_NO_WAIT(1);
#endif

#endif
}
#endif

#undef USE_LDS_CHILDREN
#undef FOLLOWER_STACK_ENTRY
#endif
