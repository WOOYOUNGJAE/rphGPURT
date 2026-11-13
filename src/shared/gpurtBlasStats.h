/*
 ***********************************************************************************************************************
 *
 *  Per-BLAS node statistics log (DeviceSettings::enableBlasStatsLog), shared between the BlasStats3_1 shader and
 *  the CPU side (src/gpurtBlasStatsLog.cpp).
 *
 *  The log buffer (CPU-visible, GPU-L2-uncached) is a ring of BLAS_STATS_RING_ENTRIES slots: first one done marker
 *  dword per slot (dense, so the monitor thread scans few cache lines), then one 64-byte entry per slot. A second,
 *  GPU-local buffer holds one 64-byte accumulator per slot, so the per-wave atomics stay in GPU L2 instead of going
 *  over PCIe. For a BLAS build the command buffer records CP writes (entry: seq, VA, flags; accumulator: zeroes;
 *  marker: 0), a barrier, then the BlasStats3_1 dispatch. Its last thread group copies the accumulated counts into
 *  the entry and writes seq into the marker, so a non-zero marker means the entry is complete.
 *
 **********************************************************************************************************************/
#ifndef _GPURT_BLAS_STATS_H
#define _GPURT_BLAS_STATS_H

#define BLAS_STATS_RING_ENTRIES             16384   // Slots in the ring
#define BLAS_STATS_ENTRY_SIZE               64      // Bytes per entry
#define BLAS_STATS_THREADGROUP_SIZE         64      // One lane per box node

// Dword indices inside an entry. The accumulator uses the same indices for the counters and GROUPS_DONE.
#define BLAS_STATS_ENTRY_SEQ                0       // CP write: sequence number (also the done marker value)
#define BLAS_STATS_ENTRY_BLAS_VA_LO         1       // CP write: BLAS API GPU VA
#define BLAS_STATS_ENTRY_BLAS_VA_HI         2
#define BLAS_STATS_ENTRY_MERGE_FLAGS        3       // CP write: WideNodeMergingFlags if this build merges, else 0
#define BLAS_STATS_ENTRY_OBB_FLAGS          4       // CP write: effective EnableOrientedBoundingBoxes of this build
#define BLAS_STATS_ENTRY_RESERVED           5
#define BLAS_STATS_ENTRY_NUM_PRIMS          6       // Shader: header numPrimitives
#define BLAS_STATS_ENTRY_POST               7       // Shader: header numInternalNodesFp32 (box nodes, incl. root)
#define BLAS_STATS_ENTRY_FOLLOWERS          8       // Shader: box slots whose previous slot is a box with range 0
#define BLAS_STATS_ENTRY_LEAF_ONLY          9       // Shader: box nodes without a box child
#define BLAS_STATS_ENTRY_DOBB               10      // Shader: box nodes with OBB matrix index != 0x7f
#define BLAS_STATS_ENTRY_DOBB_LEAF_ONLY     11      // Shader: leaf-only box nodes with an OBB
#define BLAS_STATS_ENTRY_NODES_SCANNED      12      // Shader: nodes visited (== POST unless the dispatch was too small)
#define BLAS_STATS_ENTRY_GROUPS_DONE        13      // Accumulator only: finished thread groups
#define BLAS_STATS_ENTRY_DWORDS             16

#define BLAS_STATS_ENTRY_CP_DWORDS          6       // Entry dwords [0, 6) written by the CP
#define BLAS_STATS_ACCUM_FIRST_DWORD        8       // Accumulator dwords [8, 14) are zeroed by the CP
#define BLAS_STATS_ACCUM_DWORDS             6

#endif
