#ifndef _COLLAPSE_AND_ENCODE_SCRATCH_BVH_HLSL
#define _COLLAPSE_AND_ENCODE_SCRATCH_BVH_HLSL

// Collapses the BVH2 into BVH8 nodes and encodes them in the RTIP3.1 hardware format, as EncodeHwBvh3_1 does.
// It can also merge sibling leaf-only boxes during the collapse (collapse-time merge, WideNodeMergingFlags 0x1).
// The merge additions are marked "(Merge)" and drop out when IsCollapseTimeMergeEnabled() is false.
//   block = sibling boxes merged into one node; leader = its first slot (gets the dst node), followers = the rest.
//   Parent P decides the merge from the leaf slot counts of its box children (same decisions as EncodeScratchBvh),
//   pushes only leaders, and writes a block info for each block. The block task pulls up each member in its own
//   leaf segment.

// Block stack entry: bit 31 set, [3:30] = stack index of P, [0:2] = leader slot in P
#define BLOCK_STACK_ENTRY_FLAG             0x80000000u
#define BLOCK_STACK_ENTRY_MAX_PARENT_INDEX ((1u << 28) - 1)

// Block info (MERGE_BLOCK_INFO_*): written by P at its stack index in bvhWideNodeData, read by the block task.
//   CHILDREN: BVH2 root of each member slot. LAYOUT: [0:7] follower slot mask, [8:31] (size - 1) per slot, 3 bits.
#define BLOCK_INFO_STRIDE          MERGE_BLOCK_INFO_STRIDE
#define BLOCK_INFO_CHILDREN_OFFSET MERGE_BLOCK_INFO_CHILDREN_OFFSET
#define BLOCK_INFO_LAYOUT_OFFSET   MERGE_BLOCK_INFO_LAYOUT_OFFSET
#define BLOCK_LAYOUT_SIZE_SHIFT    8
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

// Only TaskMacros.hlsl uses SharedMem (task index broadcast)
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
// (Merge) Whether this pass merges: a merging triangle BLAS build (IsLeafSlotCountEnabled), with or without the leaf
// compression pass. Without it, each leaf slot gets one inline primitive structure, which also works for block nodes.
// Procedural BLAS do not merge.
bool IsCollapseTimeMergeEnabled()
{
#if (MERGE_BOX_NODES_FLAG == 2)
    return IsLeafSlotCountEnabled() && (IsTrianglePrimitiveBuild() || IsCompressedTrianglePrimitiveBuild());
#else
    return false; // merging needs MERGE_BOX_NODES_FLAG == 2
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
// (Merge) Same node load as the EncodeHwBvh3_1 PullUp loop
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
// (Merge) PullUp of all block members at once. Member m opens boxes in EncodeHwBvh3_1's PullUp order within its own
// lane segment [start, end), 4 bits per member in segmentStarts / segmentEnds.
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

            // Open a box only if the segment has both a box and an empty lane
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
// EncodeHwBvh3_1's EncodeCompressedBoxNode plus merge. A block task pulls up the members of a block instead of a
// BVH2 node. Lanes stay as PullUp filled them, and each lane writes its child to childSlot.
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
    const bool nodeMergeEnabled = IsCollapseTimeMergeEnabled();

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

    // (Merge) Changed for a block task
    uint parentScratchNodeIdx = stackEntrySrc; // writes the prim comp batch (the leader for a block)
    uint numFollowers         = 0;             // followers of a block task, 0 otherwise
    uint leafStartMask        = 0;             // bit k: a member's leaf segment starts at leaf k (OBB table)

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
        scratchNode = (ScratchNode)0;

        // (Merge) Read the block info P wrote at its stack index
        const uint parentStackIdx  = (stackEntrySrc & ~BLOCK_STACK_ENTRY_FLAG) >> 3;
        const uint leaderSlot      = stackEntrySrc & bits(3);
        const uint blockInfoOffset = ShaderConstants.offsets.bvhWideNodeData + (parentStackIdx * BLOCK_INFO_STRIDE);

        // members[k]: BVH2 root of member k
        const uint4 members     =
            ScratchGlobal.Load4(blockInfoOffset + BLOCK_INFO_CHILDREN_OFFSET + (leaderSlot * sizeof(uint)));
        const uint  blockLayout = ScratchGlobal.Load(blockInfoOffset + BLOCK_INFO_LAYOUT_OFFSET);

        // The block runs from the leader slot through the follower slots right after it
        const uint followerSlotMask = blockLayout & bits(LANE_GROUP_SIZE);
        const uint numMembers       = firstbitlow(~followerSlotMask & ~bits(leaderSlot + 1)) - leaderSlot;
        GPU_ASSERT((numMembers >= 2) && (numMembers <= BLOCK_MAX_MEMBERS));

        // Member k's root goes to the first lane of its segment
        uint segmentStarts  = 0;
        uint segmentEnds    = 0;
        uint numBlockLeaves = 0;
        uint memberSize     = 0;
        [unroll]
        for (uint k = 0; k < BLOCK_MAX_MEMBERS; ++k)
        {
            if (k < numMembers)
            {
                const uint sizeK = ((blockLayout >> (BLOCK_LAYOUT_SIZE_SHIFT + (3 * (leaderSlot + k)))) & bits(3)) + 1;
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

        numFollowers         = numMembers - 1;
        parentScratchNodeIdx = members.x;

        // Load the member roots. The block's box is the union of the member boxes.
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
            memberBbox = GetScratchNodeBoundingBoxTS(scratchNodesScratchOffset, false, scratchNode);
            GPU_ASSERT(isBox && (FetchScratchNodeLeafSlotCount(scratchNode, false) == memberSize));
        }
        parentBbox.min = laneGroup.Min(memberBbox.min);
        parentBbox.max = laneGroup.Max(memberBbox.max);

        // Members are leaf-only, so each segment fills exactly
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

    // (Merge) Merge decision from the box child sizes. childSlot: HW slot of this lane's child. dstBoxChildIdx: index
    // among this node's dst box nodes (leaders only). Without merging both are the EncodeHwBvh3_1 values.
    uint childSlot         = isLeaf ? (boxOrLeafChildIndex + numBoxChildren) : boxOrLeafChildIndex;
    uint dstBoxChildIdx    = boxOrLeafChildIndex;
    uint numDstBoxChildren = numBoxChildren;
    uint leaderSlotMask    = bits(numBoxChildren);
    uint followerSlotMask  = 0;
    uint packedSizes       = 0;     // 4 bits per slot
    bool isDstBoxChild     = isBox; // leader
    bool isBlockLeader     = false; // leader with followers

    if (nodeMergeEnabled && (numBoxChildren >= 2))
    {
        // Box child size: its leaf slot count if leaf-only, else 8
        uint boxSize = 0;
        if (isBox)
        {
            boxSize = min(FetchScratchNodeLeafSlotCount(scratchNode, false), 8u);
            GPU_ASSERT(boxSize >= 2);
        }
        packedSizes = laneGroup.BitOr(isBox ? (boxSize << (4 * childSlot)) : 0u);

        followerSlotMask = ComputeMergeFollowerMask(packedSizes, numBoxChildren);

        leaderSlotMask    = bits(numBoxChildren) & ~followerSlotMask;
        numDstBoxChildren = countbits(leaderSlotMask);
        isDstBoxChild     = isBox && ((leaderSlotMask & (1u << childSlot)) != 0);
        dstBoxChildIdx    = countbits(leaderSlotMask & bits(childSlot));
        isBlockLeader     = isDstBoxChild && ((followerSlotMask & (1u << (childSlot + 1))) != 0);
    }
    const bool hasBlock = (followerSlotMask != 0);

    // (Merge) Write the block info at this node's stack index, before the dst allocation atomic
    if (hasBlock)
    {
        const uint blockInfoOffset = ShaderConstants.offsets.bvhWideNodeData + (bvhNodeStackIdx * BLOCK_INFO_STRIDE);
        const bool isBlockFollower = isBox && ((followerSlotMask & (1u << childSlot)) != 0);

        if (isBlockLeader || isBlockFollower)
        {
            ScratchGlobal.Store(blockInfoOffset + BLOCK_INFO_CHILDREN_OFFSET + (childSlot * sizeof(uint)),
                                scratchNodeIdx);
        }

        if (laneGroup.IsFirstLane())
        {
            uint blockLayout = followerSlotMask;
            [unroll]
            for (uint k = 0; k < LANE_GROUP_SIZE; ++k)
            {
                if (k < numBoxChildren)
                {
                    blockLayout |= (GetPackedSlotSize(packedSizes, k) - 1) << (BLOCK_LAYOUT_SIZE_SHIFT + (3 * k));
                }
            }
            ScratchGlobal.Store(blockInfoOffset + BLOCK_INFO_LAYOUT_OFFSET, blockLayout);
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

            // (Merge) Only leaders get a dst node and a stack entry
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
        // (Merge) The block task reads the block info after it sees its stack entry
        DeviceMemoryBarrier();
    }

    // Write box nodes to the global memory to be process by another thread
    if (isDstBoxChild)
    {
        uint childStackIdx = dstStackIdx + dstBoxChildIdx;
        uint packedDstNodeIdx = PackStackDstNodeIdx(baseDstBoxIdx + dstBoxChildIdx, dstBoxChildIdx);

        // (Merge) Block entry: this node's stack index and the leader slot
        uint childStackEntrySrc = scratchNodeIdx;
        if (isBlockLeader)
        {
            GPU_ASSERT(bvhNodeStackIdx < BLOCK_STACK_ENTRY_MAX_PARENT_INDEX);
            childStackEntrySrc = BLOCK_STACK_ENTRY_FLAG | (bvhNodeStackIdx << 3) | childSlot;
        }

        ScratchGlobal.Store2(ShaderConstants.offsets.qbvhGlobalStack + (childStackIdx * sizeof(uint2)),
                         uint2(childStackEntrySrc, packedDstNodeIdx));
    }

    baseLeafNodeOffset = laneGroup.ReadFirstLane(baseLeafNodeOffset);
    baseBoxNodeOffset  = laneGroup.ReadFirstLane(baseBoxNodeOffset);

    uint basePrimStructIdx = 0;
    if (laneGroup.IsFirstLane())
    {
        // (Merge) A block's box was computed before PullUp
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
        uint obbMatrixIndexBits = PackObbMatrixIndexBits(INVALID_OBB, packedSourceBoxes);

        // (Merge) Allow-update builds also store a block node's leaf start mask in bits [31:24] (0 for other nodes), so
        // Update3_1 can refit each member's parent slot from its own leaf segment. These builds have no OBBs, so the
        // OBB refit's 16-bit write of bits [15:0] never runs on them.
        if (IsUpdateAllowed())
        {
            GPU_ASSERT(leafStartMask <= bits(QUANTIZED_BVH8_NODE_LEAF_START_MASK_BITS));
            obbMatrixIndexBits |= leafStartMask << QUANTIZED_BVH8_NODE_LEAF_START_MASK_SHIFT;
        }
        QBVH8::WriteObbMatrixIndex(currNodeOffset, obbMatrixIndexBits);

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

        // (Merge) Range length 0 when the next slot is not a leader, so a follower points to its leader's node
        uint boxRangeLength = 1;
        if (nodeMergeEnabled)
        {
            boxRangeLength = ((leaderSlotMask & (1u << (childSlot + 1))) == 0) ? 0 : 1; // leader nodes are consecutive
        }

        const uint primRangeLength = isLeaf ? (EnableLeafCompressionPass() ? 0 : 1) : boxRangeLength;
        const uint baseNodeOffset  = isLeaf ? baseLeafNodeOffset : baseBoxNodeOffset;
        const uint offset          = childSlot;

        // (Merge) Used for leaders and leaves only. A follower's dstBoxChildIdx counts its own leader, so this would be
        // the next leader's node (HW finds the leader's node from the range lengths above).
        const uint childNodeOffset =
            baseNodeOffset + ((isLeaf ? boxOrLeafChildIndex : dstBoxChildIdx) * QBVH8::NodeSizeInBytes);

        // (Merge) Members keep their own bounds
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

        // (Loosen DOBB, deprecated) Pick the OBB child with each block's surface area split over its members.
        // Without LOOSEN_DOBB these equal the original values.
        uint  loosenObbChildIdx = aabbMaxChildIdx;
        uint  loosenObbLane     = aabbMaxLane;
        float loosenAabbSum     = aabbSum;
#if LOOSEN_DOBB
        if (IsLoosenDobbEnabled())
        {
            uint numBlockMembers = 1;
            if (isBox)
            {
                // Member count of this lane's block
                const uint notFollowerMask = ~followerSlotMask & bits(LANE_GROUP_SIZE);
                const uint slotAndBelow    = bits(childSlot + 1);
                const uint blockLeaderSlot = firstbithigh(notFollowerMask & slotAndBelow);
                const uint nextLeaderSlot  = firstbitlow((notFollowerMask | (1u << LANE_GROUP_SIZE)) & ~slotAndBelow);
                numBlockMembers = nextLeaderSlot - blockLeaderSlot;
                GPU_ASSERT(numBlockMembers > 0);
            }

            const float loosenSa  = aabbSa / float(numBlockMembers);
            const float loosenMax = laneGroup.Max(loosenSa);
            loosenAabbSum     = laneGroup.Sum(loosenSa);
            loosenObbLane     = laneGroup.GetFirstActiveLaneIndex(loosenSa == loosenMax);
            loosenObbChildIdx = laneGroup.Broadcast(childSlot, loosenObbLane);
        }
#endif

        // (Best4) With merging, merged leaf-only nodes take their leaves' matrices (collectObbCandidates). Other
        // nodes take their 4 largest children's (collectChildObbCandidates): leaf-only nodes without merging, every
        // node with Best4 on all nodes.
        const bool collectObbCandidates = IsObbBest4Enabled() && IsCollapseTimeMergeEnabled() &&
                                          (numBoxChildren == 0) && (numFollowers != 0);
        const bool collectChildObbCandidates =
            IsObbBest4Enabled() && (collectObbCandidates == false) &&
            (IsObbBest4AllNodes() || ((IsCollapseTimeMergeEnabled() == false) && (numBoxChildren == 0)));

        uint bestObbMatrixIdx = INVALID_OBB;
        if (((aabbMaxLane == laneGroup.laneIndex) || (loosenObbLane == laneGroup.laneIndex) ||
             collectObbCandidates || collectChildObbCandidates) && (isLeaf))
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

        const uint leafObbMatrixIdx = bestObbMatrixIdx; // this lane's leaf matrix (when computed above)
        bestObbMatrixIdx = laneGroup.Broadcast(bestObbMatrixIdx, aabbMaxLane);

        // (Loosen DOBB) leaf matrix of the loosened best child
        uint loosenObbMatrixIdx = bestObbMatrixIdx;
#if LOOSEN_DOBB
        if (IsLoosenDobbEnabled())
        {
            loosenObbMatrixIdx = laneGroup.Broadcast(leafObbMatrixIdx, loosenObbLane);
        }
#endif

        // (Best4) Candidates for the refit (OBB_TABLE_CANDIDATES_OFFSET); candidate 0 is the current choice
        uint obbCandidates = OBB_CANDIDATES_ALL_EMPTY;
        if (collectObbCandidates)
        {
            // (Merge) Merged leaf-only node: the next largest leaves' matrices, without duplicates
            obbCandidates = bitFieldInsert(obbCandidates, 0, 8, loosenObbMatrixIdx);

            bool isUnlisted = isLeaf && (leafObbMatrixIdx != INVALID_OBB) && (leafObbMatrixIdx != loosenObbMatrixIdx);
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
            // As in EncodeHwBvh3_1: a box child is named by its child slot (= its refit lane)
            const uint laneSlot = isLeaf ? leafObbMatrixIdx : (OBB_CANDIDATE_BOX_LANE_BIT | childSlot);
            obbCandidates = bitFieldInsert(obbCandidates, 0, 8, laneGroup.Broadcast(laneSlot, loosenObbLane));

            // (Merge) All members of a block get the same matrix, so a block takes one candidate (childBlock)
            const uint childBlock = IsCollapseTimeMergeEnabled() ?
                                    firstbithigh(~followerSlotMask & bits(childSlot + 1)) : laneGroup.laneIndex;
            const uint firstPicked = IsCollapseTimeMergeEnabled() ? laneGroup.Broadcast(childBlock, loosenObbLane) :
                                                                  loosenObbLane;

            bool isUnpicked = (isLeaf || isBox) && (childBlock != firstPicked) &&
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
                    const uint picked = IsCollapseTimeMergeEnabled() ? laneGroup.Broadcast(childBlock, lane) : lane;
                    isUnpicked    = isUnpicked && (childBlock != picked);
                }
            }
        }

        if (laneGroup.IsFirstLane())
        {
            if (numBoxChildren == 0)
            {
                uint obbRefitStackIndex = 0;

                ScratchGlobal.InterlockedAdd(
                    ShaderConstants.offsets.obbRefitStackPtrs + STACK_PTRS_OBB_REFIT_BATCH_COUNT, 1, obbRefitStackIndex);

                // (Merge) Plain node index; the block layout is in the leaf start mask of the OBB flags
                ScratchGlobal.Store(
                    ShaderConstants.offsets.obbRefitStack + (obbRefitStackIndex * sizeof(uint)), bvhNodeDstIdx);
            }

            // OBB refit flags are packed as follows:
            // b[0:3]   : 2's complement of box child count. The maximum value can be 8. That requires 3 bits plus the sign
            //           bit --> 4 bits. The maximum carry bits (for when box count is incremented by refit kernel)
            //           is 1, but we have plenty more room in unused bits beside.
            // b[4:13]  : 2's complement increment overflow bits
            // b[14:21] : Leaf start mask of a block node (bit k: a member's leaf segment starts at leaf k), else 0
            // b[22:24] : Child index with maximum AABB surface area for this node.
            // b[25:31] : OBB matrix index of the best leaf node amongst the children. This is precomputed and stored in
            //           scratch node during morton code generation phase to absorb the VGPR impact of that computation.
            // (Merge) b[0:3] counts leaders only. b[22:31] hold the loosened best child (the original one without
            // Loosen DOBB).
            const uint encodedBits = (loosenObbMatrixIdx << 25u) |
                                     (loosenObbChildIdx  << 22u) |
                                     (leafStartMask      << OBB_FLAGS_BIT_OFFSET_LEAF_START_MASK) |
                                     ((numDstBoxChildren ^ 0xFu) + 1);

            if (IsObbTableExtended())
            {
                // 16 byte entry (see OBB_TABLE_EXTENDED_ENTRY_SIZE): loosened best child and surface area sum, then the
                // original best child and the candidates
                const uint origBest = (bestObbMatrixIdx << 3) | aabbMaxChildIdx;
                const uint obbFlagsOffset =
                    ShaderConstants.offsets.qbvhObbFlags + (bvhNodeDstIdx * OBB_TABLE_EXTENDED_ENTRY_SIZE);
                ScratchGlobal.Store<uint4>(obbFlagsOffset,
                                           uint4(encodedBits, asuint(loosenAabbSum), origBest, obbCandidates));
            }
            else
            {
                const uint obbFlagsOffset =
                    ShaderConstants.offsets.qbvhObbFlags + (bvhNodeDstIdx * OBB_TABLE_ENTRY_SIZE);
                ScratchGlobal.Store<uint2>(obbFlagsOffset, uint2(encodedBits, asuint(aabbSum)));
            }
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
// EncodeHwBvhImpl plus merging when IsCollapseTimeMergeEnabled()
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
            // (Merge) BVH2 node or block entry
            const uint bvhNodeSrcIdx = stackEntry.x;
            const bool isBlockTask   = IsCollapseTimeMergeEnabled() && ((stackEntry.x & BLOCK_STACK_ENTRY_FLAG) != 0);
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
                                    globalStackIndex,
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

    PostHwBvhBuild(localId, numActivePrims, Gfx12::IsCollapseTimeMergeEnabled());

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
#undef BLOCK_INFO_STRIDE
#undef BLOCK_INFO_CHILDREN_OFFSET
#undef BLOCK_INFO_LAYOUT_OFFSET
#undef BLOCK_LAYOUT_SIZE_SHIFT
#undef BLOCK_MAX_MEMBERS
#endif
