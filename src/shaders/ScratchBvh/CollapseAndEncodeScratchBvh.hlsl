#ifndef _COLLAPSE_AND_ENCODE_SCRATCH_BVH_HLSL
#define _COLLAPSE_AND_ENCODE_SCRATCH_BVH_HLSL

// (SinglePass, WideNodeMergingFlags 0x1) This pass is EncodeHwBvh3_1 plus leaf-only box merge.
//   With one pass merging (0x1 without 0x2), every BLAS uses this pass instead of EncodeHwBvh3_1. Merge only happens on BLASes where the builder enabled merge (not updatable)
//   (IsSinglePassMergeEnabled, spec constant). With merge off, only the same code as EncodeHwBvh3_1 remains.
//   Removed from EncodeHwBvh3_1: legacy leaf compression (BUILD_PARALLEL), rebraid.
//
// Merge part:
//   Terms: block = sibling boxes combined into one node, member = a box belonging to a block (both leader and follower),
//          leader = the member with the earliest slot (receives the dst node), follower = a member that is not the leader.
//   The BVH2 build (MergeScratchNodes) records for each internal node its wide leaf count (leaf count after full wide collapse, clamped to 9).
//   So when collapsing parent P, the sizes of its box children are already known and the merge can be decided on the spot.
//   1. Wide-collapse P (same PullUp as EncodeHwBvh3_1).
//   2. Pack the box child sizes as 4 bits per slot and run the contiguous greedy (same decision as EncodeScratchBvh).
//   3. Only leaders receive a dst node and get pushed (MERGE_BOX_NODES_FLAG == 2).
//      A leader with followers (a block) is pushed as an entry pointing to the descriptor P wrote at its own stack index.
//   4. A block task runs PullUp for each member within its own leaf range (segment). All segments are opened together.
//      The leaf layout is the same as 2-pass (follower leaves appended after the leader's leaves).
// The resulting topology is the same as 2-pass with the same WNMF. What may differ:
//   - dst node numbering order
//   - OBB table: when several children have the same AABB SA, it picks the child with the earlier PullUp lane like EncodeHwBvh3_1 (2-pass picks the
//     child with the earlier slot). The SA sum (aabbSum) is also accumulated in lane order like EncodeHwBvh3_1. Either way the table stores that child's slot.

// If bit 31 of stack entry .x is set, it is a block task (INVALID_IDX is filtered out first). BVH2 indices never use bit 31.
//   [0:2]  : slot of the block leader within parent P
//   [3:30] : stack index of parent P. The descriptor is stored at this location
#define BLOCK_STACK_ENTRY_FLAG             0x80000000u
#define BLOCK_STACK_ENTRY_MAX_PARENT_INDEX ((1u << 28) - 1) // parent stack index must be less than this (does not collide with INVALID_IDX)

// block descriptor: parent P writes it into scratch bvhWideNodeData at P's stack index. In 1-pass this region holds only descriptors,
// not ScratchWideNodes. The builder sizes it as internal node count x BLOCK_DESC_STRIDE (SINGLE_PASS_BLOCK_DESC_*,
// shared/rayTracingDefs.h). The task knows its stack index from the start, so it can be written before the dst allocation atomic.
//   [0, 32)                : BVH2 root index of the box child per slot. Only slots belonging to a block (leader, follower) are written
//   BLOCK_DESC_INFO_OFFSET : [0:7] follower slot mask (slots merged into an earlier sibling), [8:31] (size - 1) per slot, 3 bits each
// Members (including the leader) have size >= 2 and total size <= 8, so there are at most 4.
#define BLOCK_DESC_STRIDE          SINGLE_PASS_BLOCK_DESC_STRIDE
#define BLOCK_DESC_CHILDREN_OFFSET SINGLE_PASS_BLOCK_DESC_CHILDREN_OFFSET
#define BLOCK_DESC_INFO_OFFSET     SINGLE_PASS_BLOCK_DESC_INFO_OFFSET
#define BLOCK_DESC_SIZE_SHIFT      8
#define BLOCK_MAX_MEMBERS          4

#if NO_SHADER_ENTRYPOINT == 0
//=====================================================================================================================
#include "../../shadersClean/common/ShaderDefs.hlsli"

#define GC_DSTBUFFER
#define GC_DSTMETADATA
#define GC_SCRATCHBUFFER
#include "../../shadersClean/build/BuildRootSignature.hlsli"

#define TASK_COUNTER_BUFFER   ScratchGlobal
#define TASK_COUNTER_OFFSET   (ShaderConstants.offsets.taskLoopCounters + TASK_LOOP_QBVH_COUNTER_OFFSET)
#define NUM_TASKS_DONE_OFFSET (ShaderConstants.offsets.taskLoopCounters + TASK_LOOP_QBVH_TASKS_DONE_OFFSET)
#include "../TaskMacros.hlsl"

template<typename T>
T LoadInstanceDescBuffer(uint offset)
{
    return InstanceDescBuffer.Load<T>(offset);
}
#include "../IndirectArgBufferUtils.hlsl"

#define LANE_GROUP_SIZE 8

// SharedMem is only used by TaskMacros.hlsl (INIT_TASK/END_TASK broadcast the task index through SharedMem[0]).
// EncodeHwBvh3_1 declares 8 DWORDs per thread for the legacy leaf compression pass, which this pass does not have.
groupshared uint SharedMem[1];
uint GetSharedMem(uint index)
{
    return SharedMem[index];
}
void SetSharedMem(uint index, uint value)
{
    SharedMem[index] = value;
}

#include "../../shadersClean/build/BuildCommonScratch.hlsli"
#endif

#if GPURT_BUILD_RTIP3_1

#include "../rtip3_1.hlsli"
#include "../EncodeHwBvhCommon3_1.hlsl"
#include "../QBVH8Common.hlsl"
#include "../../shadersClean/common/LaneGroup.hlsli"
#include "../PrimitiveStructureEncoder3_1.hlsl"
#include "../OrientedBoundingBoxes.hlsl"
#include "MergePack.hlsli"

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
// (SinglePass) Whether this pass performs merge. The builder writes the merging flags to build settings only for BLAS builds
// that merge, and ENABLE without LEGACY_TWO_PASS only for those that merge in this pass, and only then does the BVH2 build record the wide leaf count used for merge. Like EncodeScratchBvh, it requires a
// wave leaf compression pass. If false, this pass does the same work as EncodeHwBvh3_1.
bool IsSinglePassMergeEnabled()
{
#if (MERGE_BOX_NODES_FLAG == 2)
    return IsWideLeafCountEnabled() && EnableWaveLeafCompressionPass();
#else
    return false; // only the compacted destination layout (MERGE_BOX_NODES_FLAG == 2) is supported for merging
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
void WriteInstanceNode(
    in AccelStructOffsets offsets,
    in uint               numPrimRefs,
    in ScratchNode        scratchNode,
    in uint               scratchNodeIndex,
    in uint               nodeOffset,
    in uint               parentNodePointer)
{
    const uint instanceIndex = scratchNode.left_or_primIndex_or_instIndex;
    const uint nodePointer = PackNodePointer(NODE_TYPE_USER_NODE_INSTANCE, nodeOffset);
    InstanceDesc instanceDesc = LoadInstanceDesc(instanceIndex, LoadNumPrimAndOffset().primitiveOffset);

    const uint blasRootNodePointer = scratchNode.splitBox_or_nodePointer;
    const uint blasMetadataSize = scratchNode.numPrimitivesAndDoCollapse;
    const uint geometryType = ExtractScratchNodeGeometryType(scratchNode.packedFlags);

    CullIllegalInstances(blasMetadataSize, scratchNode, instanceDesc);

    WriteInstanceNode3_1(instanceDesc,
                         geometryType,
                         instanceIndex,
                         nodePointer,
                         blasRootNodePointer,
                         blasMetadataSize,
                         offsets,
                         parentNodePointer);

    // The destination index is the instance index.
    const uint destIndex = instanceIndex;

    DstBuffer.Store(offsets.primNodePtrs + (destIndex * sizeof(uint)), nodePointer);
}

//=====================================================================================================================
static void WriteLeafNode(
    in uint               scratchNodesScratchOffset,
    in AccelStructOffsets offsets,
    in uint               numPrimRefs,
    in ScratchNode        scratchNode,
    in uint               scratchNodeIdx,
    in uint               primStructIdx,
    in uint               nodeOffset,
    in uint               parentNodePointer)
{
    if (Settings.topLevelBuild)
    {
        Gfx12::WriteInstanceNode(offsets,
                                 numPrimRefs,
                                 scratchNode,
                                 scratchNodeIdx,
                                 nodeOffset,
                                 parentNodePointer);
    }
    else
    {
        bool isTri1Valid = false;
        bool isTri0Opaque = false;
        bool isTri1Opaque = false;
        uint primId0 = 0;
        uint primId1 = 0;
        uint geomId0 = 0;
        uint geomId1 = 0;
        TriangleData tri0 = (TriangleData) 0;
        TriangleData tri1 = (TriangleData) 0;

        const bool isQuadPrim =
            Settings.enableEarlyPairCompression && IsScratchNodeQuadPrimitive(scratchNode);

        if (isQuadPrim)
        {
            isTri1Valid = true;
            isTri0Opaque = IsOpaqueNode(scratchNode.packedFlags);
            isTri1Opaque = isTri0Opaque;
            primId0 = GetScratchNodePrimitiveIndex(scratchNode, 0);
            primId1 = GetScratchNodePrimitiveIndex(scratchNode, 1);
            geomId0 = ExtractScratchNodeGeometryIndex(scratchNode);
            geomId1 = geomId0;
            tri0 = GetScratchNodeQuadVertices(scratchNodesScratchOffset, scratchNodeIdx, 0);
            tri1 = GetScratchNodeQuadVertices(scratchNodesScratchOffset, scratchNodeIdx, 1);
        }
        else if (IsLeafNode(scratchNodeIdx, numPrimRefs) == false)
        {
            uint firstNodeIndex;
            ScratchNode scratchNodeTri0;

            uint secondNodeIndex;
            ScratchNode scratchNodeTri1;

            if (UsePrimIndicesArray())
            {
                firstNodeIndex = GetPrimRefIdx(FetchSortedPrimIndex(ShaderConstants.offsets.primIndicesSorted,
                                                                    scratchNode.sortedPrimIndex),
                                               numPrimRefs);
                scratchNodeTri0 = FetchScratchNode(scratchNodesScratchOffset, firstNodeIndex);

                secondNodeIndex = GetPrimRefIdx(FetchSortedPrimIndex(ShaderConstants.offsets.primIndicesSorted,
                                                                     scratchNode.sortedPrimIndex + 1),
                                                numPrimRefs);
                scratchNodeTri1 = FetchScratchNode(scratchNodesScratchOffset, secondNodeIndex);
            }
            else if (CollapseAnyPairs())
            {
                firstNodeIndex = scratchNode.left_or_primIndex_or_instIndex;
                scratchNodeTri0 = FetchScratchNode(scratchNodesScratchOffset, firstNodeIndex);

                secondNodeIndex = scratchNode.right_or_geometryIndex;
                scratchNodeTri1 = FetchScratchNode(scratchNodesScratchOffset, secondNodeIndex);
            }
            else
            {
                firstNodeIndex = scratchNode.left_or_primIndex_or_instIndex;
                scratchNodeTri0 = FetchScratchNode(scratchNodesScratchOffset, firstNodeIndex);

                secondNodeIndex = scratchNodeTri0.parent;
                scratchNodeTri1 = FetchScratchNode(scratchNodesScratchOffset, secondNodeIndex);
            }

            isTri1Valid = true;
            isTri0Opaque = IsOpaqueNode(scratchNodeTri0.packedFlags);
            isTri1Opaque = IsOpaqueNode(scratchNodeTri1.packedFlags);
            primId0 = scratchNodeTri0.left_or_primIndex_or_instIndex;
            primId1 = scratchNodeTri1.left_or_primIndex_or_instIndex;
            geomId0 = ExtractScratchNodeGeometryIndex(scratchNodeTri0);
            geomId1 = ExtractScratchNodeGeometryIndex(scratchNodeTri1);
            tri0.v0 = scratchNodeTri0.bbox_min_or_v0;
            tri0.v1 = scratchNodeTri0.bbox_max_or_v1;
            tri0.v2 = scratchNodeTri0.sah_or_v2_or_instBasePtr;
            tri1.v0 = scratchNodeTri1.bbox_min_or_v0;
            tri1.v1 = scratchNodeTri1.bbox_max_or_v1;
            tri1.v2 = scratchNodeTri1.sah_or_v2_or_instBasePtr;
        }
        else
        {
            isTri0Opaque = IsOpaqueNode(scratchNode.packedFlags);
            primId0 = scratchNode.left_or_primIndex_or_instIndex;
            geomId0 = ExtractScratchNodeGeometryIndex(scratchNode);
            tri0.v0 = scratchNode.bbox_min_or_v0;
            tri0.v1 = scratchNode.bbox_max_or_v1;
            tri0.v2 = scratchNode.sah_or_v2_or_instBasePtr;
        }

        PrimStruct3_1::WritePairPrimStruct(tri0,
                                           isTri0Opaque,
                                           primId0,
                                           geomId0,
                                           isTri1Valid,
                                           tri1,
                                           isTri1Opaque,
                                           primId1,
                                           geomId1,
                                           nodeOffset);

        DstBuffer.Store(offsets.primNodePtrs + (primStructIdx * sizeof(uint)), nodeOffset);
    }
}

//=====================================================================================================================
static uint3 ComputePackedChildInfo(
    in uint          scratchNodesScratchOffset,
    in ScratchNode   childScratchNode,
    in uint          childScratchNodeIdx,
    in uint          nodeType,
    in uint          nodeRangeLength,
    in float3        rcpExponents,
    in float3        origin,
    in uint          numPrimRefs)
{
    const bool isLeafNode = IsLeafNode(childScratchNodeIdx, numPrimRefs);
    BoundingBox childBounds = GetScratchNodeBoundingBoxTS(scratchNodesScratchOffset, isLeafNode, childScratchNode);

    const uint instanceMask =
        ExtractScratchNodeInstanceMask(childScratchNode.packedFlags);

    const uint boxNodeFlags =
        ExtractScratchNodeBoxFlags(childScratchNode.packedFlags);

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
// Box splitting / Child reuse
static void BoxSplitting(
    LaneGroup                laneGroup,
    in    uint               scratchNodesScratchOffset,
    in    uint               currNodeOffset,
    in    uint               indexInParent,
    in    AccelStructOffsets offsets,
    in    uint               baseLeafNodeOffset,
    in    uint               maxNumChildren,
    in    uint               numLeafChildren,
    in    uint               numBoxChildren,
    in    uint               boxOrLeafChildIndex,
    in    ScratchNode        childScratchNode,
    in    uint               numEmptySlots,
    in    float3             rcpExponents,
    in    uint3              exponents,
    in    BoundingBox        parentBbox,
    in    bool               isLeaf)
{
    uint maxNodeLane = 0;

    if (Settings.topLevelBuild && (numEmptySlots > 0) && (Settings.boxSplittingFlags & BoxSplittingFlags::Instance))
    {
        const NumPrimAndOffset numPrimAndOffset = LoadNumPrimAndOffset();

        // Make sure the compressed box node is encoded
        DeviceMemoryBarrier();

        // Pick the instances in decreasing SAH order
        if (isLeaf)
        {
            // Flag to indicate if the instance node in the current lane has been processed.
            bool processed = false;

            // Note, boxOrLeafChildIndex is only the index among the leaf children here.
            uint childInfoIndex = numBoxChildren + boxOrLeafChildIndex;

            BoundingBox instanceBounds[4];

            const uint intersectableInstanceOffset =
                baseLeafNodeOffset + (boxOrLeafChildIndex * QBVH8::NodeSizeInBytes) + sizeof(HwInstanceTransformNode);
            const uint4 dw = DstBuffer.Load4(intersectableInstanceOffset);

            // Fetch origin from intersectable instance node
            const float3 instanceOrigin = float3(asfloat(dw.xyz));

            // Decode valid child count from intersectable instance node
            const uint numValidInstanceBounds = ExtractBvh4ValidChildCount(dw.w);

            // Extract packed exponents from intersectable instance node
            uint3 instanceExponents;
            instanceExponents.x = bitFieldExtract(dw.w, 0, 8);
            instanceExponents.y = bitFieldExtract(dw.w, 8, 8);
            instanceExponents.z = bitFieldExtract(dw.w, 16, 8);

            // Decode intersectable instance bounds
            for (uint j = 0; j < 4; ++j)
            {
                const uint childInfoOffset = intersectableInstanceOffset + GetBvh4ChildInfoOffset(j);
                const ChildInfo childInfo = DstBuffer.Load<ChildInfo>(childInfoOffset);
                if (childInfo.Valid())
                {
                    instanceBounds[j] = childInfo.DecodeBounds(instanceOrigin, instanceExponents);
                }
                else
                {
                    instanceBounds[j] = InvalidBoundingBox;
                }
            }

            uint numFinalBounds = numValidInstanceBounds;

            const float area =
                ComputeBoxSurfaceArea(GetScratchNodeBoundingBoxTS(scratchNodesScratchOffset, true, childScratchNode));

            while ((numEmptySlots > 0) && (processed == false))
            {
                const float maxArea = laneGroup.Max(area);
                const bool isMaxAreaNode = (area == maxArea);
                maxNodeLane = laneGroup.GetFirstActiveLaneIndex(isMaxAreaNode); // Get the lane index of the max node

                // Do box splitting on the instance node with max area.
                if (laneGroup.laneIndex == maxNodeLane)
                {
                    // Mark this instance node as processed.
                    processed = true;

                    // Merge intersectable instance bounds if there are not enough empty slots available.
                    // TODO: Intersectable instance bounds sorted in increasing order of SAH would be ideal for
                    //       merging here

                    if ((numEmptySlots == 1) && (numValidInstanceBounds == 4))
                    {
                        // Merge [2:3]
                        instanceBounds[2] = CombineAABB(instanceBounds[2], instanceBounds[3]);

                        numFinalBounds--;
                    }

                    if (((numEmptySlots == 1) && (numValidInstanceBounds >= 3)) ||
                        ((numEmptySlots == 2) && (numValidInstanceBounds == 4)))
                    {
                        // Merge [0:1]
                        instanceBounds[0] = CombineAABB(instanceBounds[0], instanceBounds[1]);

                        numFinalBounds--;

                        // Copy [2:3] to [1:2] to remove gaps
                        instanceBounds[1] = instanceBounds[2];
                        instanceBounds[2] = instanceBounds[3];
                    }

                    // Make space for the box splitting child infos
                    for (uint j = maxNumChildren - numEmptySlots - 1; j > childInfoIndex; --j)
                    {
                        const uint srcChildInfoOffset = currNodeOffset + GetChildInfoOffset(j);
                        const uint dstChildInfoOffset = currNodeOffset + GetChildInfoOffset(j + numFinalBounds - 1);
                        const uint3 childInfo = DstBuffer.Load3(srcChildInfoOffset);
                        DstBuffer.Store3(dstChildInfoOffset, childInfo);
                    }

                    const uint instanceIndex = childScratchNode.left_or_primIndex_or_instIndex;
                    const uint instanceMask = ExtractScratchNodeInstanceMask(childScratchNode.packedFlags);
                    const uint boxNodeFlags = ExtractScratchNodeBoxFlags(childScratchNode.packedFlags);

                    const InstanceDesc instanceDesc = LoadInstanceDesc(instanceIndex, numPrimAndOffset.primitiveOffset);

                    // Apply objectToWorld transform to intersectable instance bounds and write to child infos
                    for (uint j = 0; j < numFinalBounds; ++j)
                    {
                        instanceBounds[j] = TransformBoundingBox(instanceBounds[j], instanceDesc.Transform);

                        // Child reuse has node range length 0
                        const uint nodeRangeLength = (j < (numFinalBounds - 1)) ? 0 : 1;

                        const uint3 childInfo = QBVH8::ComputePackedChildInfo(instanceBounds[j],
                                                                              instanceMask,
                                                                              boxNodeFlags,
                                                                              NODE_TYPE_USER_NODE_INSTANCE,
                                                                              nodeRangeLength,
                                                                              rcpExponents,
                                                                              parentBbox.min);

                        QBVH8::WritePackedChildInfo(currNodeOffset, childInfoIndex + j, childInfo);
                    }

                    numEmptySlots -= numFinalBounds - 1;
                }

                const uint maxNodeChildInfoIndex = laneGroup.Broadcast(childInfoIndex, maxNodeLane);
                const uint maxNodeNumFinalBounds = laneGroup.Broadcast(numFinalBounds, maxNodeLane);

                // Update any child index that were shifted over to make space above.
                if (childInfoIndex > maxNodeChildInfoIndex)
                {
                    childInfoIndex += maxNodeNumFinalBounds - 1;
                }

                numEmptySlots = laneGroup.Broadcast(numEmptySlots, maxNodeLane);

                // Update the child count on the last iteration.
                if (((numEmptySlots == 0) || (laneGroup.BallotCount(processed == false) == 0)) &&
                    laneGroup.IsFirstActiveLane())
                {
                    QBVH8::WritePackedExpChildIdxAndCount(currNodeOffset,
                                                          exponents,
                                                          indexInParent,
                                                          maxNumChildren - numEmptySlots);
                }
            }
        }
    }
}

//=====================================================================================================================
// Allocation of compressed leaf nodes is deferred to the compression pass.
uint GetNumInlinePrimStructs(
    LaneGroup   laneGroup,
    uint        scratchNodesScratchOffset,
    uint        numPrimRefs,
    uint        numLeafChildren,
    uint        childScratchNodeIdx,
    bool        isLeaf)
{
    uint numInlinePrimStructs = 0;

    if (EnableWaveLeafCompressionPass())
    {
        numInlinePrimStructs = 0;
    }
    else if (Settings.maxPrimRangeSize <= 2)
    {
        numInlinePrimStructs = numLeafChildren;
    }
    else
    {
        uint numPrimStructs = 0;
        if (isLeaf)
        {
            const uint rangeNumPrims = FetchScratchNodeNumPrimitives(
                scratchNodesScratchOffset, childScratchNodeIdx, IsLeafNode(childScratchNodeIdx, numPrimRefs));

            // Minimum two triangles per prim struct in each range
            numPrimStructs = RoundUpQuotient(rangeNumPrims, 2);
        }
        numInlinePrimStructs = laneGroup.Sum(numPrimStructs);
    }

    return numInlinePrimStructs;
}

//=====================================================================================================================
// (SinglePass, block task) Node read for PullUp. Same as the load in the EncodeHwBvh3_1 PullUp loop.
static void LoadPullUpNode(
    in  uint        scratchNodesScratchOffset,
    in  uint        numPrimRefs,
    in  uint        scratchNodeIdx,
    out ScratchNode scratchNode,
    out bool        isBox,
    out bool        isLeaf,
    out float       scratchNodeArea)
{
    scratchNode = FetchScratchNode(scratchNodesScratchOffset, scratchNodeIdx);

    bool isCollapse = scratchNode.numPrimitivesAndDoCollapse & 1;

    if (CollapseAnyPairs())
    {
        isCollapse = (scratchNode.numPrimitivesAndDoCollapse >> 1) == 2;
    }

    isLeaf = (IsLeafNode(scratchNodeIdx, numPrimRefs) ||
              ((Settings.topLevelBuild == false) && isCollapse));
    isBox  = !isLeaf;
    scratchNodeArea = isLeaf ? -1.0f : scratchNode.sah_or_v2_or_instBasePtr.y;
}

//=====================================================================================================================
// (SinglePass, block task) Each member opens boxes within its own segment lanes in the same order as EncodeHwBvh3_1's PullUp
// (open the box with the largest SA, ties go to the earlier lane; the right child goes to the first empty lane in the segment).
// Segments do not overlap, so in each iteration every segment opens one box together. The number of load waits becomes
// the largest member size instead of the sum of member sizes, and the leaf layout is the same as opening members one by one.
//   segmentStarts, segmentEnds: lane range [start, end) of member m, 4 bits each. There are numSegments members (at most BLOCK_MAX_MEMBERS)
static void PullUpSegments(
    LaneGroup         laneGroup,
    in    uint        scratchNodesScratchOffset,
    in    uint        numPrimRefs,
    in    uint        numSegments,
    in    uint        segmentStarts,
    in    uint        segmentEnds,
    inout uint        scratchNodeIdx,
    inout ScratchNode scratchNode,
    inout bool        needToLoad,
    inout bool        isBox,
    inout bool        isLeaf,
    inout float       scratchNodeArea)
{
    while (1)
    {
        // Load node data if needed
        if (needToLoad)
        {
            LoadPullUpNode(scratchNodesScratchOffset,
                           numPrimRefs,
                           scratchNodeIdx,
                           scratchNode,
                           isBox,
                           isLeaf,
                           scratchNodeArea);

            needToLoad = false;
        }

        const uint boxLaneMask  = laneGroup.Ballot(isBox);
        const uint freeLaneMask = laneGroup.Ballot(scratchNodeIdx == INVALID_IDX);

        bool isOpened = false; // uniform within the lane group

        [unroll]
        for (uint m = 0; m < BLOCK_MAX_MEMBERS; ++m)
        {
            const uint segmentMask =
                (m < numSegments) ? (bits((segmentEnds >> (4 * m)) & 0xFu) & ~bits((segmentStarts >> (4 * m)) & 0xFu)) :
                                    0u;
            const bool isInSegment = ((segmentMask >> laneGroup.laneIndex) & 1u) != 0;

            // Only segments that have both a box to open and an empty lane open one (a leaf-only member fills its segment exactly)
            const bool doOpen = ((boxLaneMask & segmentMask) != 0) && ((freeLaneMask & segmentMask) != 0);

            const float maxArea      = laneGroup.Max(isInSegment ? scratchNodeArea : -1.0f);
            const bool isMaxAreaNode = isInSegment && (scratchNodeArea == maxArea);
            // Get the lane index of the max node in this segment
            const uint maxNodeLane      = laneGroup.GetFirstActiveLaneIndex(isMaxAreaNode);
            const uint firstInvalidLane = firstbitlow(freeLaneMask & segmentMask);

            uint rightNode = INVALID_IDX;

            if (doOpen && (laneGroup.laneIndex == maxNodeLane))
            {
                // Assign left child for next processing
                scratchNodeIdx = scratchNode.left_or_primIndex_or_instIndex;
                rightNode = scratchNode.right_or_geometryIndex;
                needToLoad = true;
            }

            // Distribute the right child to the first thread with an invalid index in this segment
            rightNode = laneGroup.Broadcast(rightNode, maxNodeLane);

            if (doOpen && (laneGroup.laneIndex == firstInvalidLane) && (rightNode != INVALID_IDX))
            {
                // This thread takes the right child
                scratchNodeIdx = rightNode;
                needToLoad = true;
            }

            isOpened = isOpened || doOpen;
        }

        if (isOpened == false)
        {
            // Break the loop if no segment has a box node left to open
            break;
        }
    }
}

//=====================================================================================================================
// EncodeHwBvh3_1's EncodeCompressedBoxNode plus merge. Merge parts are marked "(SinglePass)", and
// when IsSinglePassMergeEnabled() is false (spec constant) they all drop out, leaving the same code as EncodeHwBvh3_1.
//   Normal task: PullUp from stackEntrySrc (a BVH2 node) (EncodeHwBvh3_1).
//   Block task: PullUp the members of the parent descriptor pointed to by stackEntrySrc, each within its own segment
//               (results in a leaf-only node).
// Lanes stay as PullUp filled them, and each lane writes to its own child's slot (childSlot).
static void EncodeCompressedBoxNode(
    LaneGroup                laneGroup,
    in    uint               rootNodeIndex,
    in    uint               scratchNodesScratchOffset,
    in    uint               stackEntrySrc,
    in    bool               isBlockTask,
    in    uint               bvhNodeStackIdx,
    in    uint               bvhNodeDstIdx,
    in    uint               currNodePtr,
    in    uint               currNodeOffset,
    in    uint               indexInParent,
    in    AccelStructOffsets offsets,
    in    uint               maxNumChildren,
    in    uint               numPrimRefs)
{
    const bool nodeMergeEnabled = IsSinglePassMergeEnabled();

    uint baseDstBoxIdx      = 0;
    uint numLeafChildren    = 0;
    uint numBoxChildren     = 0;
    uint baseLeafNodeOffset = 0;
    uint baseBoxNodeOffset  = 0;
    float3 rcpExponents     = 0;
    uint3 exponents;
    BoundingBox parentBbox;

    bool isBox      = false;
    bool needToLoad = false;
    bool isLeaf     = false;
    uint boxOrLeafChildIndex;
    uint scratchNodeIdx  = INVALID_IDX; // Start with an invalid index for each thread.
    float scratchNodeArea    = -1.0f;   // Mark area as invalid initially.
    ScratchNode scratchNode;

    // This bitfield is an array of 2 bit indices indicating which of the first 4 boxes was split to produce a given
    // box. It is initialized to [0, 1, 2, 3] for the first 4 boxes. After the first 4 boxes are generated, each
    // additional box takes the index of the box it was split from.
    uint packedSourceBoxes = 0b11100100;

    // (SinglePass) The values below change for a block task
    uint parentScratchNodeIdx = stackEntrySrc; // BVH2 node that writes the prim comp batch (the leader for a block)
    uint leafStartMask        = 0;             // for a block: bits at the leaf positions where member segments start (OBB table)

    if (isBlockTask == false)
    {
        // Initialization for the first lane
        if (laneGroup.IsFirstLane())
        {
            scratchNodeIdx = parentScratchNodeIdx;
            needToLoad = true;
        }

        while (1)
        {
            const uint firstInvalidLane = laneGroup.GetFirstActiveLaneIndex(scratchNodeIdx == INVALID_IDX);

            // Load node data if needed
            if (needToLoad)
            {
                scratchNode = FetchScratchNode(scratchNodesScratchOffset, scratchNodeIdx);

                bool isCollapse = scratchNode.numPrimitivesAndDoCollapse & 1;

                if (CollapseAnyPairs())
                {
                    isCollapse = (scratchNode.numPrimitivesAndDoCollapse >> 1) == 2;
                }

                isLeaf = (IsLeafNode(scratchNodeIdx, numPrimRefs) ||
                          ((Settings.topLevelBuild == false) && isCollapse));
                isBox  = !isLeaf;
                scratchNodeArea = isLeaf ? -1.0f : scratchNode.sah_or_v2_or_instBasePtr.y;

                needToLoad = false;
            }

            numBoxChildren = laneGroup.BallotCount(isBox);

            if ((numBoxChildren == 0) || (firstInvalidLane >= maxNumChildren))
            {
                // Break the loop if there are no more box nodes needing to be processed,
                // or if there's no remaining space in the structure.
                break;
            }

            const float maxArea      = laneGroup.Max(scratchNodeArea);
            const bool isMaxAreaNode = (scratchNodeArea == maxArea);
            // Get the lane index of the max node
            const uint maxNodeLane   = laneGroup.GetFirstActiveLaneIndex(isMaxAreaNode);

            uint rightNode = INVALID_IDX;

            if (laneGroup.laneIndex == maxNodeLane)
            {
                // Assign left child for next processing
                scratchNodeIdx = scratchNode.left_or_primIndex_or_instIndex;
                rightNode = scratchNode.right_or_geometryIndex;
                needToLoad = true;
            }

            // Distribute the right child to the first thread with an invalid index
            rightNode = laneGroup.Broadcast(rightNode, maxNodeLane);

            if ((laneGroup.laneIndex == firstInvalidLane) && (rightNode != INVALID_IDX))
            {
                // This thread takes the right child
                scratchNodeIdx = rightNode;
                needToLoad = true;

                if (laneGroup.laneIndex >= 4)
                {
                    const uint sourceBoxIndex = (packedSourceBoxes >> (maxNodeLane * 2)) & 0b11;
                    packedSourceBoxes |= sourceBoxIndex << (laneGroup.laneIndex * 2);
                }
            }
        }
    }
    else
    {
        scratchNode = (ScratchNode)0; // LoadPullUpNode / PullUpSegments take it as out / inout

        // (SinglePass) Read the descriptor that parent P wrote at its own stack index in one go (all lanes read the same dword)
        const uint parentStackIdx = (stackEntrySrc & ~BLOCK_STACK_ENTRY_FLAG) >> 3;
        const uint leaderSlot     = stackEntrySrc & bits(3);
        const uint descOffset     = ShaderConstants.offsets.bvhWideNodeData + (parentStackIdx * BLOCK_DESC_STRIDE);

        // root of member k = members[k] (slots continuing from the leader slot)
        const uint4 members   = ScratchGlobal.Load4(descOffset + BLOCK_DESC_CHILDREN_OFFSET + (leaderSlot * sizeof(uint)));
        const uint  blockInfo = ScratchGlobal.Load(descOffset + BLOCK_DESC_INFO_OFFSET);

        // This block spans from the slot after the leader as far as the follower slots continue
        const uint followerSlotMask = blockInfo & bits(LANE_GROUP_SIZE);
        const uint numMembers       = firstbitlow(~followerSlotMask & ~bits(leaderSlot + 1)) - leaderSlot;
        GPU_ASSERT((numMembers >= 2) && (numMembers <= BLOCK_MAX_MEMBERS));

        // segment start = sum of leaf counts of the preceding members (sizes are in the descriptor). A member root is placed in the first lane of its segment.
        uint segmentStarts  = 0;
        uint segmentEnds    = 0;
        uint numBlockLeaves = 0;
        uint memberSize     = 0; // first lane of a segment: the size of that member
        [unroll]
        for (uint k = 0; k < BLOCK_MAX_MEMBERS; ++k)
        {
            if (k < numMembers)
            {
                const uint sizeK = ((blockInfo >> (BLOCK_DESC_SIZE_SHIFT + (3 * (leaderSlot + k)))) & bits(3)) + 1;
                GPU_ASSERT(sizeK >= 2);

                if (laneGroup.laneIndex == numBlockLeaves)
                {
                    scratchNodeIdx = members[k];
                    needToLoad     = true;
                    memberSize     = sizeK;
                }

                segmentStarts  |= numBlockLeaves << (4 * k);
                leafStartMask  |= 1u << numBlockLeaves; // bit 0 is always the leader
                numBlockLeaves += sizeK;
                segmentEnds    |= numBlockLeaves << (4 * k);
            }
        }
        GPU_ASSERT(numBlockLeaves <= LANE_GROUP_SIZE);

        parentScratchNodeIdx = members.x;

        // Read the member roots (the first load of PullUp). Block box = union of member boxes (same as the bbox update in 2-pass merge)
        if (needToLoad)
        {
            LoadPullUpNode(scratchNodesScratchOffset,
                           numPrimRefs,
                           scratchNodeIdx,
                           scratchNode,
                           isBox,
                           isLeaf,
                           scratchNodeArea);

            needToLoad = false;
        }

        BoundingBox memberBbox;
        memberBbox.min = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        memberBbox.max = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        if (scratchNodeIdx != INVALID_IDX)
        {
            // A member is a box child of the parent, so it is never a BVH2 leaf.
            memberBbox = GetScratchNodeBoundingBoxTS(scratchNodesScratchOffset, false, scratchNode);
            GPU_ASSERT(isBox && (FetchScratchNodeWideLeafCount(scratchNode, false) == memberSize));
        }
        parentBbox.min = laneGroup.Min(memberBbox.min);
        parentBbox.max = laneGroup.Max(memberBbox.max);

        // All members run PullUp together, each within its own segment. Since they are leaf-only, each segment fills exactly.
        PullUpSegments(laneGroup,
                       scratchNodesScratchOffset,
                       numPrimRefs,
                       numMembers,
                       segmentStarts,
                       segmentEnds,
                       scratchNodeIdx,
                       scratchNode,
                       needToLoad,
                       isBox,
                       isLeaf,
                       scratchNodeArea);
        GPU_ASSERT(laneGroup.Ballot(isBox) == 0);
        GPU_ASSERT(laneGroup.Ballot(isLeaf) == bits(numBlockLeaves));

        numBoxChildren = laneGroup.BallotCount(isBox);
    }
    numLeafChildren = laneGroup.BallotCount(isLeaf);

    // Calculate exclusive prefix sum to get index (0,1,2... )
    if (isBox)
    {
        boxOrLeafChildIndex = laneGroup.ExclusivePrefixSumBool(true);
    }
    else if (isLeaf)
    {
        boxOrLeafChildIndex = laneGroup.ExclusivePrefixSumBool(true);
    }

    // (SinglePass) Merge decision (looks only at box child sizes)
    //   childSlot     : HW slot of this lane's child (boxes first, then leaves).
    //   dstBoxChildIdx: index of the box child among this node's dst box nodes. Only leaders receive dst nodes, laid out contiguously.
    // Without merge these equal EncodeHwBvh3_1's values (childSlot = box/leaf order, dst = box order, every box child gets a dst).
    uint childSlot         = isLeaf ? (boxOrLeafChildIndex + numBoxChildren) : boxOrLeafChildIndex;
    uint dstBoxChildIdx    = boxOrLeafChildIndex;
    uint numDstBoxChildren = numBoxChildren;
    uint leaderSlotMask    = bits(numBoxChildren);
    uint followerSlotMask  = 0;     // follower slots (slots merged into an earlier sibling)
    uint packedSizes       = 0;     // box child sizes, 4 bits per slot
    bool isDstBoxChild     = isBox; // box child that receives a dst node (leader)
    bool isBlockLeader     = false; // leader with followers

    if (nodeMergeEnabled && (numBoxChildren >= 2))
    {
        // size = numBoxes + numLeaves. A leaf-only box uses its wide leaf count; any other box fills all 8 slots.
        // packedSizes is uniform within the lane group, so the decision is pure ALU with no cross-lane exchange.
        uint boxSize = 0;
        if (isBox)
        {
            boxSize = min(FetchScratchNodeWideLeafCount(scratchNode, false), 8u);
            GPU_ASSERT(boxSize >= 2);
        }
        packedSizes = laneGroup.BitOr(isBox ? (boxSize << (4 * childSlot)) : 0u);

        followerSlotMask = ComputeMergeFollowerMask(packedSizes, numBoxChildren);

        leaderSlotMask    = bits(numBoxChildren) & ~followerSlotMask;
        numDstBoxChildren = countbits(leaderSlotMask);
        isDstBoxChild     = isBox && ((leaderSlotMask & (1u << childSlot)) != 0);
        dstBoxChildIdx    = countbits(leaderSlotMask & bits(childSlot));
        // followers occupy the slots right after the leader
        isBlockLeader     = isDstBoxChild && ((followerSlotMask & (1u << (childSlot + 1))) != 0);
    }
    const bool hasBlock = (followerSlotMask != 0);

    // (SinglePass) block descriptor: written at this node's (P's) stack index. The address is known from the start, so it can be written before the dst
    // allocation atomic, and the wait at the barrier before push overlaps that atomic's round trip. Box lanes in a block write their own slots directly.
    if (hasBlock)
    {
        const uint descOffset = ShaderConstants.offsets.bvhWideNodeData + (bvhNodeStackIdx * BLOCK_DESC_STRIDE);
        const bool isBlockFollower = isBox && ((followerSlotMask & (1u << childSlot)) != 0);

        if (isBlockLeader || isBlockFollower)
        {
            ScratchGlobal.Store(descOffset + BLOCK_DESC_CHILDREN_OFFSET + (childSlot * sizeof(uint)), scratchNodeIdx);
        }

        if (laneGroup.IsFirstLane())
        {
            uint blockInfo = followerSlotMask;
            [unroll]
            for (uint k = 0; k < LANE_GROUP_SIZE; ++k)
            {
                if (k < numBoxChildren)
                {
                    blockInfo |= (GetPackedSlotSize(packedSizes, k) - 1) << (BLOCK_DESC_SIZE_SHIFT + (3 * k));
                }
            }
            ScratchGlobal.Store(descOffset + BLOCK_DESC_INFO_OFFSET, blockInfo);
        }
    }

    uint dstStackIdx = 0;

    if (Settings.topLevelBuild == 0)
    {
        const uint numInlinePrimStructs = GetNumInlinePrimStructs(laneGroup,
                                                                  scratchNodesScratchOffset,
                                                                  numPrimRefs,
                                                                  numLeafChildren,
                                                                  scratchNodeIdx,
                                                                  isLeaf);

        if (laneGroup.IsFirstLane())
        {
            IncrementAccelStructHeaderField(ACCEL_STRUCT_HEADER_NUM_INTERNAL_FP32_NODES_OFFSET, 1);

            // This code assumes STACK_PTRS_SRC_PTR_OFFSET and STACK_PTRS_DST_PTR_OFFSET are in consecutive
            // DWORDs.
            const uint stackPtrOffset = ShaderConstants.offsets.qbvhGlobalStackPtrs + STACK_PTRS_SRC_PTR_OFFSET;

            // (SinglePass) followers receive neither a dst node nor a stack entry (numDstBoxChildren = number of leaders)
            uint64_t returnVal;
            ScratchGlobal.InterlockedAdd(stackPtrOffset, numDstBoxChildren, dstStackIdx);
            ScratchGlobal.InterlockedAdd(stackPtrOffset + 4, numDstBoxChildren + numInlinePrimStructs, baseDstBoxIdx);

            baseBoxNodeOffset = offsets.internalNodes + (baseDstBoxIdx * QBVH8::NodeSizeInBytes);

            // BLAS leaf nodes are allocated in a unified data section unlike TLAS which requires the instance nodes
            // to be in a separate data section for sideband addressing. See ComputeInstanceSidebandOffset()
            baseLeafNodeOffset = EnableWaveLeafCompressionPass() ?
                0 : (baseBoxNodeOffset + (numDstBoxChildren * QBVH8::NodeSizeInBytes));
        }
    }
    else
    {
        if (laneGroup.IsFirstLane())
        {
            // TODO: This atomic increment can be removed if we place fp32 node count and leaf node count
            // adjacent in memory and use a single 64-bit atomic to increment both box and leaf node counts
            IncrementAccelStructHeaderField(ACCEL_STRUCT_HEADER_NUM_INTERNAL_FP32_NODES_OFFSET, 1);

            // This code assumes STACK_PTRS_SRC_PTR_OFFSET and STACK_PTRS_DST_PTR_OFFSET are in consecutive
            // DWORDs.
            const uint stackPtrOffset = ShaderConstants.offsets.qbvhGlobalStackPtrs + STACK_PTRS_SRC_PTR_OFFSET;

            uint baseDstLeafIdx = 0;
            uint64_t returnVal;
            ScratchGlobal.InterlockedAdd(stackPtrOffset, numBoxChildren, baseDstBoxIdx);
            ScratchGlobal.InterlockedAdd(stackPtrOffset + 4, numLeafChildren, baseDstLeafIdx);
            dstStackIdx = baseDstBoxIdx;

            baseBoxNodeOffset = offsets.internalNodes + (baseDstBoxIdx * QBVH8::NodeSizeInBytes);
            baseLeafNodeOffset = offsets.leafNodes + (baseDstLeafIdx * QBVH8::NodeSizeInBytes);
        }
    }

    baseDstBoxIdx = laneGroup.ReadFirstLane(baseDstBoxIdx);
    dstStackIdx   = laneGroup.ReadFirstLane(dstStackIdx);

    if (hasBlock)
    {
        // (SinglePass) The block task reads the descriptor after it sees its stack entry
        DeviceMemoryBarrier();
    }

    // Write box nodes to the global memory to be process by another thread
    if (isDstBoxChild)
    {
        uint childStackIdx = dstStackIdx + dstBoxChildIdx;
        uint packedDstNodeIdx = PackStackDstNodeIdx(baseDstBoxIdx + dstBoxChildIdx, dstBoxChildIdx);

        // (SinglePass) a block finds its descriptor from the parent's (this node's) stack index and the leader slot
        uint childStackEntrySrc = scratchNodeIdx;
        if (isBlockLeader)
        {
            GPU_ASSERT(bvhNodeStackIdx < BLOCK_STACK_ENTRY_MAX_PARENT_INDEX);
            childStackEntrySrc = BLOCK_STACK_ENTRY_FLAG | (bvhNodeStackIdx << 3) | childSlot;
        }

        // Regular operation for other threads with isBox set
        ScratchGlobal.Store2(ShaderConstants.offsets.qbvhGlobalStack + (childStackIdx * sizeof(uint2)),
                         uint2(childStackEntrySrc, packedDstNodeIdx));
    }

    baseLeafNodeOffset = laneGroup.ReadFirstLane(baseLeafNodeOffset);
    baseBoxNodeOffset  = laneGroup.ReadFirstLane(baseBoxNodeOffset);

    uint basePrimStructIdx = 0;
    if (laneGroup.IsFirstLane())
    {
        // (SinglePass) the block's box was computed before PullUp as the union of member boxes
        if (isBlockTask == false)
        {
            const ScratchNode parentScratchNode = FetchScratchNode(scratchNodesScratchOffset, parentScratchNodeIdx);

            // Note, the current subtree being collapsed can be just a single primitive
            const bool isLeafNode = IsLeafNode(parentScratchNodeIdx, numPrimRefs);
            parentBbox = GetScratchNodeBoundingBoxTS(scratchNodesScratchOffset, isLeafNode, parentScratchNode);
        }

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
        else if (Settings.topLevelBuild == 0)
        {
            QBVH8::WriteLeafNodeBaseOffset(currNodeOffset, INVALID_NODE);
        }

        QBVH8::WriteOrigin(currNodeOffset, parentBbox.min);

        const uint validChildCount = numBoxChildren + numLeafChildren;
        QBVH8::WritePackedExpChildIdxAndCount(currNodeOffset, exponents, indexInParent, validChildCount);

        // Store the source box bitfield in the upper bits of the OBB DWORD (unused by HW). This is used to construct
        // four filter boxes for instances.
        QBVH8::WriteObbMatrixIndex(currNodeOffset, PackObbMatrixIndexBits(INVALID_OBB, packedSourceBoxes));

        if ((Settings.topLevelBuild == false) && (EnableLeafCompressionPass() == false))
        {
            basePrimStructIdx = IncrementAccelStructHeaderField(ACCEL_STRUCT_HEADER_NUM_LEAF_NODES_OFFSET, numLeafChildren);
        }
    }

    parentBbox.min    = laneGroup.ReadFirstLane(parentBbox.min);
    parentBbox.max    = laneGroup.ReadFirstLane(parentBbox.max);
    exponents         = laneGroup.ReadFirstLane(exponents);
    rcpExponents      = laneGroup.ReadFirstLane(rcpExponents);
    basePrimStructIdx = laneGroup.ReadFirstLane(basePrimStructIdx);

    if (scratchNodeIdx != INVALID_IDX)
    {
        // TLAS builds initialize these fields immediately. They should be left 0 for BLAS builds if we are compressing
        // leaf nodes. If we are not compressing, a single triangle primitive structure is produced immediately.
        // Either way the triangle node type field is 0.
        const uint nodeType        = isLeaf ? (Settings.topLevelBuild ?
                                               NODE_TYPE_USER_NODE_INSTANCE : NODE_TYPE_TRIANGLE_0) :
                                               NODE_TYPE_BOX_QUANTIZED_BVH8;

        // (SinglePass, same as EncodeScratchBvh) During the decode stage of traverse, nodeRangeLength indicates how
        // many internal nodes away the next sibling is from this boxNode. If the right sibling is a follower,
        // increment it by 0 so that it points to the same node as the left sibling.
        uint boxRangeLength = 1;
        if (nodeMergeEnabled)
        {
            boxRangeLength = ((leaderSlotMask & (1u << (childSlot + 1))) == 0) ? 0 : 1; // leader nodes are consecutive
        }

        const uint primRangeLength = isLeaf ? (EnableLeafCompressionPass() ? 0 : 1) : boxRangeLength;
        const uint baseNodeOffset  = isLeaf ? baseLeafNodeOffset : baseBoxNodeOffset;
        const uint offset          = childSlot;

        // (SinglePass) followers also point to the leader's dst node
        const uint childNodeOffset =
            baseNodeOffset + ((isLeaf ? boxOrLeafChildIndex : dstBoxChildIdx) * QBVH8::NodeSizeInBytes);

        // (SinglePass) Box children keep their own (pre-merge) bounds in this node, also when they are block members
        const uint3 childInfo = ComputePackedChildInfo(scratchNodesScratchOffset,
                                                       scratchNode,
                                                       scratchNodeIdx,
                                                       nodeType,
                                                       primRangeLength,
                                                       rcpExponents,
                                                       parentBbox.min,
                                                       numPrimRefs);

        QBVH8::WritePackedChildInfo(currNodeOffset, offset, childInfo);

        if (isBox)
        {
            if (isDstBoxChild)
            {
                QBVH8::WriteParentPointer(childNodeOffset, currNodePtr);
            }
        }
        else //isLeaf
        {
            // Process and compress leaf children
            if (EnableLeafCompressionPass() == false)
            {
                WriteLeafNode(scratchNodesScratchOffset,
                              offsets,
                              numPrimRefs,
                              scratchNode,
                              scratchNodeIdx,
                              basePrimStructIdx + boxOrLeafChildIndex,
                              childNodeOffset,
                              currNodePtr);
            }
        }
    }
    else
    {
        QBVH8::WriteInvalidChildInfo(currNodeOffset, laneGroup.laneIndex);
    }

    if (IsObbEnabled())
    {
        float aabbSa = 0.0f;
        if (scratchNodeIdx != INVALID_IDX)
        {
            aabbSa = GetScratchNodeSurfaceArea(
                scratchNodesScratchOffset, scratchNodeIdx, scratchNode, numPrimRefs);
        }

        const float aabbSum = laneGroup.Sum(aabbSa);
        const float aabbMax = laneGroup.Max(aabbSa);

        const uint aabbMaxLane     = laneGroup.GetFirstActiveLaneIndex(aabbSa == aabbMax);
        const uint aabbMaxChildIdx = laneGroup.Broadcast(childSlot, aabbMaxLane);

        uint bestObbMatrixIdx = INVALID_OBB;
        if ((aabbMaxLane == laneGroup.laneIndex) && (isLeaf))
        {
            uint packedFlags = scratchNode.packedFlags;

            bool isCollapse = scratchNode.numPrimitivesAndDoCollapse & 1;

            if (CollapseAnyPairs())
            {
                isCollapse = (scratchNode.numPrimitivesAndDoCollapse >> 1) == 2;
            }

            if (isCollapse)
            {
                // Triangles in a range are colocated and tend to share a similar orientation, so
                // choosing one at random from the range is about as effective as choosing one with
                // a more sophisticated heuristic.
                //
                // Pick the first triangle in the range as an approximation.
                uint triNodeIdx = scratchNode.left_or_primIndex_or_instIndex;

                if (UsePrimIndicesArray())
                {
                    triNodeIdx = GetPrimRefIdx(FetchSortedPrimIndex(ShaderConstants.offsets.primIndicesSorted,
                                                                    scratchNode.sortedPrimIndex), numPrimRefs);
                }

                packedFlags =
                    FetchScratchNodeData<uint>(scratchNodesScratchOffset, triNodeIdx, SCRATCH_NODE_FLAGS_OFFSET);
            }

            bestObbMatrixIdx = ExtractScratchNodeObbMatrixIdx(packedFlags);
        }

        bestObbMatrixIdx = laneGroup.Broadcast(bestObbMatrixIdx, aabbMaxLane);

        if (laneGroup.IsFirstLane())
        {
            if (numBoxChildren == 0)
            {
                uint obbRefitStackIndex = 0;

                ScratchGlobal.InterlockedAdd(
                    ShaderConstants.offsets.obbRefitStackPtrs + STACK_PTRS_OBB_REFIT_BATCH_COUNT, 1, obbRefitStackIndex);

                // (SinglePass) The stack entry is a plain node index. How a block is partitioned is described by the
                // leaf start mask in the OBB flags below, so nothing has to be packed in here.
                ScratchGlobal.Store(
                    ShaderConstants.offsets.obbRefitStack + (obbRefitStackIndex * sizeof(uint)), bvhNodeDstIdx);
            }

            // OBB refit flags are packed as follows: (Rph Fixed)
            // b[0:3]   : 2's complement of box child count. The maximum value can be 8. That requires 3 bits plus the sign
            //           bit --> 4 bits. The maximum carry bits (for when box count is incremented by refit kernel)
            //           is 1, but we have plenty more room in unused bits beside.
            // b[4:13]  : 2's complement increment overflow bits
            // b[14:21] : data for Merged DOBB
            // b[22:24] : Child index with maximum AABB surface area for this node.
            // b[25:31] : OBB matrix index of the best leaf node amongst the children. This is precomputed and stored in
            //           scratch node during morton code generation phase to absorb the VGPR impact of that computation.
            // (SinglePass, same as EncodeScratchBvh) b[0:3] counts the box children that got a dst node (leaders),
            // and b[14:21] is the leaf start mask (bit k: a block member's leaf segment starts at leaf k, 0: not a block).
            // Without merging these are the values above.
            const uint encodedBits = (bestObbMatrixIdx << 25u) |
                                     (aabbMaxChildIdx  << 22u) |
                                     (leafStartMask      << OBB_FLAGS_BIT_OFFSET_LEAF_START_MASK) |
                                     ((numDstBoxChildren ^ 0xFu) + 1);

            const uint obbFlagsOffset = ShaderConstants.offsets.qbvhObbFlags + (bvhNodeDstIdx * OBB_TABLE_ENTRY_SIZE);
            ScratchGlobal.Store<uint2>(obbFlagsOffset, uint2(encodedBits, asuint(aabbSum)));
        }
    }

    uint localPrimCount = 0;
    if (scratchNodeIdx != INVALID_IDX)
    {
        if (isBox == false)
        {
            localPrimCount = FetchScratchNodeNumPrimitives(scratchNode, IsLeafNode(scratchNodeIdx, numPrimRefs));
        }
    }

    const uint primCount = WaveActiveSum(localPrimCount);
    if (WaveIsFirstLane())
    {
        if ((Settings.topLevelBuild == false) && (primCount > 0))
        {
            ScratchGlobal.InterlockedAdd(ShaderConstants.offsets.qbvhGlobalStackPtrs +
                                         NumLeafsDoneOffset(),
                                         primCount);
        }
    }

    const uint numEmptySlots = maxNumChildren - (numBoxChildren + numLeafChildren);

    BoxSplitting(laneGroup,
                 scratchNodesScratchOffset,
                 currNodeOffset,
                 indexInParent,
                 offsets,
                 baseLeafNodeOffset,
                 maxNumChildren,
                 numLeafChildren,
                 numBoxChildren,
                 boxOrLeafChildIndex,
                 scratchNode,
                 numEmptySlots,
                 rcpExponents,
                 exponents,
                 parentBbox,
                 isLeaf);

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
                                scratchNodeIdx,
                                boxOrLeafChildIndex,
                                isLeaf);
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
// Convert source BVH2 (in scratch memory) into RTIP3.1 compatible hardware format (EncodeHwBvhImpl), merging leaf-only
// boxes when IsSinglePassMergeEnabled()
void CollapseAndEncodeScratchBvhImpl(
    uint globalId,
    uint numPrimRefs)
{
    const AccelStructOffsets offsets = ShaderConstants.header.offsets;

    const uint rootNodeIndex = GetBvh2RootNodeIndex();

    const uint scratchNodesScratchOffset = CalculateBvhNodesOffset(ShaderConstants, numPrimRefs);

    const bool isSinglePrimBvh = (numPrimRefs == 1);

    const ScratchNode rootScratchNode = FetchScratchNode(scratchNodesScratchOffset, rootNodeIndex);
    const uint numActivePrims =
        Settings.topLevelBuild ? numPrimRefs : FetchScratchNodeNumPrimitives(rootScratchNode, isSinglePrimBvh);

    const uint maxNumInternalNodes = GetNumInternalNodeCount(numActivePrims);

    if (IsObbEnabled())
    {
        if (globalId < maxNumInternalNodes)
        {
            ScratchGlobal.Store<uint2>(ShaderConstants.offsets.qbvhObbFlags + globalId * OBB_TABLE_ENTRY_SIZE, 0);
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

        if (Settings.topLevelBuild == false)
        {
            // Generate intersectable instance node data in BLAS metadata
            if ((Settings.instanceMode == InstanceMode::Passthrough) || isSinglePrimBvh)
            {
                const uint instanceMask =
                    ExtractScratchNodeInstanceMask(rootScratchNode.packedFlags);

                WriteInstancePassthrough(CreateRootNodePointer(),
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
        }

        // Write parent pointer for root node.
        QBVH8::WriteParentPointer(offsets.internalNodes, INVALID_IDX);
    }

    LaneGroup laneGroup;
    laneGroup.Alloc(LANE_GROUP_SIZE);

    // Each stack item stores data for writes to linear QBVH memory, indexed by stack index
    const uint globalStackIndex = globalId / LANE_GROUP_SIZE;

    // With early pairing enabled, numPrimRefs represents the active primitive references. However,
    // we need to use the active primitive count to estimate the hardware internal node count.
    bool isGlobalStackEntryDone = (globalStackIndex >= maxNumInternalNodes);

    const uint maxNumChildren = 8;

    while (isGlobalStackEntryDone == false)
    {
        uint numLeafsDone = 0;
        if (WaveIsFirstLane())
        {
            numLeafsDone = ScratchGlobal.Load(ShaderConstants.offsets.qbvhGlobalStackPtrs +
                                              NumLeafsDoneOffset());
        }

        numLeafsDone = WaveReadLaneFirst(numLeafsDone);

        if (numLeafsDone >= numActivePrims)
        {
            break;
        }

        uint2 stackEntry = uint2(0, 0);
        if (laneGroup.IsFirstLane())
        {
            stackEntry = StackPop(globalStackIndex);
        }

        stackEntry = laneGroup.ReadFirstLane(stackEntry);

        // If we have a valid node on the stack, process node
        if (stackEntry.x != INVALID_IDX)
        {
            // (SinglePass) BVH2 node, or the block entry (parent stack index and leader slot)
            const uint bvhNodeSrcIdx = stackEntry.x;
            const bool isBlockTask   = IsSinglePassMergeEnabled() && ((stackEntry.x & BLOCK_STACK_ENTRY_FLAG) != 0);
            const uint bvhNodeDstIdx = GetStackDstIdx(stackEntry.y);
            const uint indexInParent = GetStackDstIdxInParent(stackEntry.y);

            const uint currNodeAddr = offsets.internalNodes + (bvhNodeDstIdx * QBVH8::NodeSizeInBytes);
            const uint currNodePtr = CreateBoxNodePointer(currNodeAddr, false);

            // Pull up the nodes chosen by SAH to be the new children in BVH8
            EncodeCompressedBoxNode(laneGroup,
                                    rootNodeIndex,
                                    scratchNodesScratchOffset,
                                    bvhNodeSrcIdx,
                                    isBlockTask,
                                    globalStackIndex, // this node's stack index (block descriptors are stored here)
                                    bvhNodeDstIdx,
                                    currNodePtr,
                                    currNodeAddr,
                                    indexInParent,
                                    offsets,
                                    maxNumChildren,
                                    numPrimRefs);

            isGlobalStackEntryDone = true;
        }
    }
}

} // namespace Gfx12

#endif

#if NO_SHADER_ENTRYPOINT == 0
//=====================================================================================================================
// Main Function : CollapseAndEncodeScratchBvh
//=====================================================================================================================
[RootSignature(RootSig)]
[numthreads(BUILD_THREADGROUP_SIZE, 1, 1)]
void CollapseAndEncodeScratchBvh(
    uint globalIdIn : SV_DispatchThreadID,
    uint groupIdIn  : SV_GroupID,
    uint localIdIn  : SV_GroupThreadID)
{
#if GPURT_BUILD_RTIP3_1
    uint globalId = globalIdIn;
    uint localId  = localIdIn;
    uint groupId  = groupIdIn;

    const uint numActivePrims = ReadAccelStructHeaderField(ACCEL_STRUCT_HEADER_NUM_ACTIVE_PRIMS_OFFSET);
    const uint maxInternalNodeCount = (numActivePrims == 0) ? 0 : GetMaxInternalNodeCount(numActivePrims);

    uint numTasksWait = 0;
    uint waveId       = 0;

    INIT_TASK;

    BEGIN_TASK(1)
    if (globalId == 0)
    {
        InitStackPtrs();
        Gfx12::InitObbRefitStackPtrs();
    }
    END_TASK(1);

    BEGIN_TASK(RoundUpQuotient(maxInternalNodeCount, BUILD_THREADGROUP_SIZE));

    ClearStackPtrs(globalId, maxInternalNodeCount, ShaderRootConstants.NumThreads());

    END_TASK(RoundUpQuotient(maxInternalNodeCount, BUILD_THREADGROUP_SIZE));

    const uint numThreads = maxInternalNodeCount * LANE_GROUP_SIZE;
    BEGIN_TASK(RoundUpQuotient(numThreads, BUILD_THREADGROUP_SIZE));

    Gfx12::CollapseAndEncodeScratchBvhImpl(globalId, numActivePrims);

    END_TASK(RoundUpQuotient(numThreads, BUILD_THREADGROUP_SIZE));

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
        }
    }

    END_TASK_NO_WAIT(1);

#endif
}
#endif

#undef BLOCK_STACK_ENTRY_FLAG
#undef BLOCK_STACK_ENTRY_MAX_PARENT_INDEX
#undef BLOCK_DESC_STRIDE
#undef BLOCK_DESC_CHILDREN_OFFSET
#undef BLOCK_DESC_INFO_OFFSET
#undef BLOCK_DESC_SIZE_SHIFT
#undef BLOCK_MAX_MEMBERS
#endif
