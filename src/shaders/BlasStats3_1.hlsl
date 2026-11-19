/*
 ***********************************************************************************************************************
 *
 *  BlasStats3_1: counts node statistics of a finished RTIP 3.1 BLAS for the BLAS statistics log
 *  (DeviceSettings::enableBlasStatsLog, see src/shared/gpurtBlasStats.h and src/gpurtBlasStatsLog.cpp).
 *
 *  RTIP 3.1 BLASes only have 128-byte QuantizedBVH8 box nodes, stored contiguously in
 *  [0, numInternalNodesFp32) from offsets.internalNodes. Each lane reads nodes globalId, globalId + numLanes, ...
 *  up to the count in the header, so indirect builds work. The CPU dispatches enough groups for the maximum node
 *  count, so in practice each lane reads at most one node.
 *
 *  Waves add their sums into a GPU-local accumulator (L2 atomics). The last thread group copies the totals and the
 *  BLAS header fields of the log into the CPU-visible ring entry and then writes the done marker.
 *
 **********************************************************************************************************************/
#include "../shadersClean/common/ShaderDefs.hlsli"
#include "../shadersClean/common/gfx12/internalNode.hlsli"
#include "../shared/gpurtBlasStats.h"

#define RootSig "RootConstants(num32BitConstants=2, b0, visibility=SHADER_VISIBILITY_ALL), "\
                "UAV(u0, visibility=SHADER_VISIBILITY_ALL), "\
                "UAV(u1, visibility=SHADER_VISIBILITY_ALL), "\
                "UAV(u2, visibility=SHADER_VISIBILITY_ALL), "\
                "UAV(u3, visibility=SHADER_VISIBILITY_ALL)"

struct BlasStatsConstants
{
    uint seq;       // Sequence number written to the done marker
    uint numGroups; // Number of dispatched thread groups
};

[[vk::push_constant]] ConstantBuffer<BlasStatsConstants> ShaderConstants : register(b0);

[[vk::binding(0, 0)]]                  RWByteAddressBuffer AccelStruct : register(u0); // BLAS header
[[vk::binding(1, 0)]] globallycoherent RWByteAddressBuffer Accum       : register(u1); // GPU-local accumulator
[[vk::binding(2, 0)]] globallycoherent RWByteAddressBuffer StatsEntry  : register(u2); // CPU-visible ring entry
[[vk::binding(3, 0)]] globallycoherent RWByteAddressBuffer DoneMarker  : register(u3); // CPU-visible done marker

//=====================================================================================================================
static void AddToAccum(uint dwordIndex, uint value)
{
    if (value != 0)
    {
        Accum.InterlockedAdd(dwordIndex * sizeof(uint), value);
    }
}

//=====================================================================================================================
static uint ReadAccum(uint dwordIndex)
{
    uint value;
    Accum.InterlockedAdd(dwordIndex * sizeof(uint), 0, value);
    return value;
}

//=====================================================================================================================
// Main Function : BlasStats3_1
//=====================================================================================================================
[RootSignature(RootSig)]
[numthreads(BLAS_STATS_THREADGROUP_SIZE, 1, 1)]
void BlasStats3_1(
    in uint3 globalThreadId : SV_DispatchThreadID,
    in uint  localId        : SV_GroupIndex)
{
    const uint numNodes            = AccelStruct.Load(ACCEL_STRUCT_HEADER_NUM_INTERNAL_FP32_NODES_OFFSET);
    const uint internalNodesOffset =
        AccelStruct.Load(ACCEL_STRUCT_HEADER_OFFSETS_OFFSET + ACCEL_STRUCT_OFFSETS_INTERNAL_NODES_OFFSET);
    const uint numLanes            = ShaderConstants.numGroups * BLAS_STATS_THREADGROUP_SIZE;

    uint numFollowers    = 0;
    uint numLeafOnly     = 0;
    uint numDobb         = 0;
    uint numDobbLeafOnly = 0;
    uint numBadBoxSlots  = 0;

    for (uint nodeIndex = globalThreadId.x; nodeIndex < numNodes; nodeIndex += numLanes)
    {
        const uint nodeOffset = internalNodesOffset + (nodeIndex * QUANTIZED_BVH8_NODE_SIZE);

        // Bits [16:23] can hold rebraid data, so only the low 7 bits are the matrix index.
        const uint obbMatrixIndex = AccelStruct.Load(nodeOffset + QUANTIZED_BVH8_NODE_OFFSET_OBB_MATRIX_INDEX) & 0x7f;

        // Same field as QuantizedBVH8BoxNode::ValidChildCount()
        const uint validChildCount =
            1 + bitFieldExtract(AccelStruct.Load(nodeOffset + QUANTIZED_BVH8_NODE_OFFSET_EXP_CHILD_IDX_AND_VALID_COUNT),
                                28,
                                3);

        bool hasBoxChild      = false;
        bool prevIsBoxRange0  = false;

        for (uint slot = 0; slot < 8; slot++)
        {
            ChildInfo childInfo =
                AccelStruct.Load<ChildInfo>(nodeOffset + QUANTIZED_BVH8_NODE_OFFSET_CHILD_INFO_0 +
                                            (slot * QUANTIZED_NODE_CHILD_INFO_STRIDE));

            const bool isValid   = childInfo.Valid();
            const bool isBoxType = (childInfo.NodeType() == NODE_TYPE_BOX_QUANTIZED_BVH8);
            const bool isBox     = isValid && isBoxType;

            // Tripwire: a used slot that points at a box node but has the invalid bounds pattern. An encoder writes
            // that pattern when the child bounds it quantizes are inverted, e.g. after reading a child node that was
            // overwritten concurrently. Such a child (and its subtree) is never hit. Expected to stay 0.
            if ((slot < validChildCount) && isBoxType && (isValid == false))
            {
                numBadBoxSlots++;
            }

            // Same rule as RefitOrientedBounds3_1: a box child whose left sibling is a box with NodeRangeLength 0
            // shares that sibling's node, so it is a follower of a merged block.
            if (isBox && prevIsBoxRange0)
            {
                numFollowers++;
            }

            prevIsBoxRange0 = isBox && (childInfo.NodeRangeLength() == 0);
            hasBoxChild     = hasBoxChild || isBox;
        }

        const uint isLeafOnly = hasBoxChild ? 0 : 1;
        const uint hasObb     = (obbMatrixIndex != ObbDisabled) ? 1 : 0;

        numLeafOnly     += isLeafOnly;
        numDobb         += hasObb;
        numDobbLeafOnly += (hasObb & isLeafOnly);
    }

    const uint waveFollowers    = WaveActiveSum(numFollowers);
    const uint waveLeafOnly     = WaveActiveSum(numLeafOnly);
    const uint waveDobb         = WaveActiveSum(numDobb);
    const uint waveDobbLeafOnly = WaveActiveSum(numDobbLeafOnly);
    const uint waveBadBoxSlots  = WaveActiveSum(numBadBoxSlots);

    if (WaveIsFirstLane())
    {
        AddToAccum(BLAS_STATS_ACCUM_FOLLOWERS,      waveFollowers);
        AddToAccum(BLAS_STATS_ACCUM_LEAF_ONLY,      waveLeafOnly);
        AddToAccum(BLAS_STATS_ACCUM_DOBB,           waveDobb);
        AddToAccum(BLAS_STATS_ACCUM_DOBB_LEAF_ONLY, waveDobbLeafOnly);
        AddToAccum(BLAS_STATS_ACCUM_BAD_BOX_SLOTS,  waveBadBoxSlots);
    }

    // Every wave of this group has finished its atomics after this.
    DeviceMemoryBarrierWithGroupSync();

    if (localId == 0)
    {
        uint prevGroupsDone = 0;
        Accum.InterlockedAdd(BLAS_STATS_ACCUM_GROUPS_DONE * sizeof(uint), 1, prevGroupsDone);

        // The last group publishes the entry: totals first, then the marker.
        if ((prevGroupsDone + 1) == ShaderConstants.numGroups)
        {
            DeviceMemoryBarrier();

            // Build flags are the low 16 bits of AccelStructHeaderInfo::flags (header info bits [16:31]).
            const uint buildFlags   =
                AccelStruct.Load(ACCEL_STRUCT_HEADER_INFO_OFFSET) >> ACCEL_STRUCT_HEADER_INFO_FLAGS_SHIFT;
            const uint geometryType = AccelStruct.Load(ACCEL_STRUCT_HEADER_GEOMETRY_TYPE_OFFSET);
            const uint buildInfo    = buildFlags | (geometryType << BLAS_STATS_BUILD_INFO_GEOMETRY_TYPE_SHIFT);

            StatsEntry.Store(BLAS_STATS_ENTRY_NUM_GEOMETRIES * sizeof(uint),
                             AccelStruct.Load(ACCEL_STRUCT_HEADER_NUM_DESCS_OFFSET));
            StatsEntry.Store(BLAS_STATS_ENTRY_NUM_PRIMS * sizeof(uint),
                             AccelStruct.Load(ACCEL_STRUCT_HEADER_NUM_PRIMS_OFFSET));
            StatsEntry.Store(BLAS_STATS_ENTRY_POST * sizeof(uint), numNodes);

            StatsEntry.Store(BLAS_STATS_ENTRY_FOLLOWERS * sizeof(uint),
                             ReadAccum(BLAS_STATS_ACCUM_FOLLOWERS));
            StatsEntry.Store(BLAS_STATS_ENTRY_LEAF_ONLY * sizeof(uint),
                             ReadAccum(BLAS_STATS_ACCUM_LEAF_ONLY));
            StatsEntry.Store(BLAS_STATS_ENTRY_DOBB * sizeof(uint),
                             ReadAccum(BLAS_STATS_ACCUM_DOBB));
            StatsEntry.Store(BLAS_STATS_ENTRY_DOBB_LEAF_ONLY * sizeof(uint),
                             ReadAccum(BLAS_STATS_ACCUM_DOBB_LEAF_ONLY));
            StatsEntry.Store(BLAS_STATS_ENTRY_BAD_BOX_SLOTS * sizeof(uint),
                             ReadAccum(BLAS_STATS_ACCUM_BAD_BOX_SLOTS));

            // RTIP 3.1: numLeafNodes counts the allocated 128-byte primitive structures (PrimStruct3_1::
            // AllocPrimStructs, BuildTrivialBvh). compactedSizeInBytes is the size the API reports for compaction
            // (CalcCompactedSize: header, box nodes, primitive structures, geometry info, primitive node pointers,
            // OBB metadata and the metadata, plus the primitive structures the CompressPrims pass adds).
            StatsEntry.Store(BLAS_STATS_ENTRY_PRIM_STRUCTS * sizeof(uint),
                             AccelStruct.Load(ACCEL_STRUCT_HEADER_NUM_LEAF_NODES_OFFSET));
            StatsEntry.Store(BLAS_STATS_ENTRY_COMPACTED_BYTES * sizeof(uint),
                             AccelStruct.Load(ACCEL_STRUCT_HEADER_COMPACTED_BYTE_SIZE_OFFSET));
            StatsEntry.Store(BLAS_STATS_ENTRY_BUILD_INFO * sizeof(uint), buildInfo);

            DeviceMemoryBarrier();

            uint unused;
            DoneMarker.InterlockedExchange(0, ShaderConstants.seq, unused);
        }
    }
}
