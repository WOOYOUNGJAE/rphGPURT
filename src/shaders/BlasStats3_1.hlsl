/*
 ***********************************************************************************************************************
 *
 *  BlasStats3_1: counts node statistics of a finished RTIP 3.1 BLAS for the BLAS statistics log
 *  (DeviceSettings::enableBlasStatsLog, see src/shared/gpurtBlasStats.h and src/gpurtBlasStatsLog.cpp).
 *
 *  RTIP 3.1 BLASes only have 128-byte QuantizedBVH8 box nodes, stored contiguously in
 *  [0, numInternalNodesFp32) from offsets.internalNodes. One lane reads one node. The CPU dispatches enough groups
 *  for the maximum node count; lanes beyond the count in the header do nothing, so indirect builds work.
 *
 *  Waves add their sums into a GPU-local accumulator (L2 atomics). The last thread group copies the totals into the
 *  CPU-visible ring entry and then writes the done marker.
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
    const uint numNodes  = AccelStruct.Load(ACCEL_STRUCT_HEADER_NUM_INTERNAL_FP32_NODES_OFFSET);
    const uint nodeIndex = globalThreadId.x;

    uint isNode        = 0;
    uint numFollowers  = 0;
    uint isLeafOnly    = 0;
    uint hasObb        = 0;

    if (nodeIndex < numNodes)
    {
        const uint internalNodesOffset =
            AccelStruct.Load(ACCEL_STRUCT_HEADER_OFFSETS_OFFSET + ACCEL_STRUCT_OFFSETS_INTERNAL_NODES_OFFSET);
        const uint nodeOffset = internalNodesOffset + (nodeIndex * QUANTIZED_BVH8_NODE_SIZE);

        // Bits [16:23] can hold rebraid data, so only the low 7 bits are the matrix index.
        const uint obbMatrixIndex = AccelStruct.Load(nodeOffset + QUANTIZED_BVH8_NODE_OFFSET_OBB_MATRIX_INDEX) & 0x7f;

        bool hasBoxChild      = false;
        bool prevIsBoxRange0  = false;

        for (uint slot = 0; slot < 8; slot++)
        {
            ChildInfo childInfo =
                AccelStruct.Load<ChildInfo>(nodeOffset + QUANTIZED_BVH8_NODE_OFFSET_CHILD_INFO_0 +
                                            (slot * QUANTIZED_NODE_CHILD_INFO_STRIDE));

            const bool isBox = childInfo.Valid() && (childInfo.NodeType() == NODE_TYPE_BOX_QUANTIZED_BVH8);

            // Same rule as RefitOrientedBounds3_1: a box child whose left sibling is a box with NodeRangeLength 0
            // shares that sibling's node, so it is a follower of a merged block.
            if (isBox && prevIsBoxRange0)
            {
                numFollowers++;
            }

            prevIsBoxRange0 = isBox && (childInfo.NodeRangeLength() == 0);
            hasBoxChild     = hasBoxChild || isBox;
        }

        isNode     = 1;
        isLeafOnly = hasBoxChild ? 0 : 1;
        hasObb     = (obbMatrixIndex != ObbDisabled) ? 1 : 0;
    }

    const uint waveNodes        = WaveActiveSum(isNode);
    const uint waveFollowers    = WaveActiveSum(numFollowers);
    const uint waveLeafOnly     = WaveActiveSum(isLeafOnly);
    const uint waveDobb         = WaveActiveSum(hasObb);
    const uint waveDobbLeafOnly = WaveActiveSum(hasObb & isLeafOnly);

    if (WaveIsFirstLane())
    {
        AddToAccum(BLAS_STATS_ENTRY_NODES_SCANNED,  waveNodes);
        AddToAccum(BLAS_STATS_ENTRY_FOLLOWERS,      waveFollowers);
        AddToAccum(BLAS_STATS_ENTRY_LEAF_ONLY,      waveLeafOnly);
        AddToAccum(BLAS_STATS_ENTRY_DOBB,           waveDobb);
        AddToAccum(BLAS_STATS_ENTRY_DOBB_LEAF_ONLY, waveDobbLeafOnly);
    }

    // Every wave of this group has finished its atomics after this.
    DeviceMemoryBarrierWithGroupSync();

    if (localId == 0)
    {
        uint prevGroupsDone = 0;
        Accum.InterlockedAdd(BLAS_STATS_ENTRY_GROUPS_DONE * sizeof(uint), 1, prevGroupsDone);

        // The last group publishes the entry: totals first, then the marker.
        if ((prevGroupsDone + 1) == ShaderConstants.numGroups)
        {
            DeviceMemoryBarrier();

            StatsEntry.Store(BLAS_STATS_ENTRY_NUM_PRIMS * sizeof(uint),
                             AccelStruct.Load(ACCEL_STRUCT_HEADER_NUM_PRIMS_OFFSET));
            StatsEntry.Store(BLAS_STATS_ENTRY_POST * sizeof(uint), numNodes);

            StatsEntry.Store(BLAS_STATS_ENTRY_FOLLOWERS * sizeof(uint),
                             ReadAccum(BLAS_STATS_ENTRY_FOLLOWERS));
            StatsEntry.Store(BLAS_STATS_ENTRY_LEAF_ONLY * sizeof(uint),
                             ReadAccum(BLAS_STATS_ENTRY_LEAF_ONLY));
            StatsEntry.Store(BLAS_STATS_ENTRY_DOBB * sizeof(uint),
                             ReadAccum(BLAS_STATS_ENTRY_DOBB));
            StatsEntry.Store(BLAS_STATS_ENTRY_DOBB_LEAF_ONLY * sizeof(uint),
                             ReadAccum(BLAS_STATS_ENTRY_DOBB_LEAF_ONLY));
            StatsEntry.Store(BLAS_STATS_ENTRY_NODES_SCANNED * sizeof(uint),
                             ReadAccum(BLAS_STATS_ENTRY_NODES_SCANNED));

            DeviceMemoryBarrier();

            uint unused;
            DoneMarker.InterlockedExchange(0, ShaderConstants.seq, unused);
        }
    }
}
