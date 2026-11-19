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
 *  marker: 0), a barrier, then the BlasStats3_1 dispatch. Its last thread group copies the accumulated counts and a
 *  few BLAS header fields into the entry and writes seq into the marker, so a non-zero marker means the entry is
 *  complete.
 *
 **********************************************************************************************************************/
#ifndef _GPURT_BLAS_STATS_H
#define _GPURT_BLAS_STATS_H

#define BLAS_STATS_RING_ENTRIES             16384   // Slots in the ring
#define BLAS_STATS_ENTRY_SIZE               64      // Bytes per entry
#define BLAS_STATS_THREADGROUP_SIZE         64      // Lanes per group; each lane loops over box nodes

// Dword indices inside a ring entry
#define BLAS_STATS_ENTRY_SEQ                0       // CP write: sequence number (also the done marker value)
#define BLAS_STATS_ENTRY_BLAS_VA_LO         1       // CP write: BLAS API GPU VA
#define BLAS_STATS_ENTRY_BLAS_VA_HI         2
#define BLAS_STATS_ENTRY_MERGE_FLAGS        3       // CP write: WideNodeMergingFlags if this build merges, else 0
#define BLAS_STATS_ENTRY_OBB_FLAGS          4       // CP write: effective EnableOrientedBoundingBoxes of this build
#define BLAS_STATS_ENTRY_NUM_GEOMETRIES     5       // Shader: header numDescs (geometry descs of the BLAS)
#define BLAS_STATS_ENTRY_NUM_PRIMS          6       // Shader: header numPrimitives
#define BLAS_STATS_ENTRY_POST               7       // Shader: header numInternalNodesFp32 (box nodes, incl. root)
#define BLAS_STATS_ENTRY_FOLLOWERS          8       // Shader: box slots whose previous slot is a box with range 0
#define BLAS_STATS_ENTRY_LEAF_ONLY          9       // Shader: box nodes without a box child
#define BLAS_STATS_ENTRY_DOBB               10      // Shader: box nodes with OBB matrix index != 0x7f
#define BLAS_STATS_ENTRY_DOBB_LEAF_ONLY     11      // Shader: leaf-only box nodes with an OBB
#define BLAS_STATS_ENTRY_BAD_BOX_SLOTS      12      // Shader: slots < ValidChildCount with box type but invalid bounds
#define BLAS_STATS_ENTRY_PRIM_STRUCTS       13      // Shader: header numLeafNodes (128-byte primitive structures)
#define BLAS_STATS_ENTRY_COMPACTED_BYTES    14      // Shader: header compactedSizeInBytes
#define BLAS_STATS_ENTRY_BUILD_INFO         15      // Shader: header build flags [0:15], geometryType [16:31]
#define BLAS_STATS_ENTRY_DWORDS             16

#define BLAS_STATS_ENTRY_CP_DWORDS          5       // Entry dwords [0, 5) are written by the CP; the shader writes
                                                    // [5, 16) before the done marker

#define BLAS_STATS_BUILD_INFO_GEOMETRY_TYPE_SHIFT 16

// Dword indices inside a slot's GPU-local accumulator (a separate 64-byte buffer, not the entry). The counters use
// the index of the entry dword they end up in. GROUPS_DONE only exists in the accumulator: entry dword 13 is
// PRIM_STRUCTS, which is unrelated.
#define BLAS_STATS_ACCUM_FOLLOWERS          BLAS_STATS_ENTRY_FOLLOWERS
#define BLAS_STATS_ACCUM_LEAF_ONLY          BLAS_STATS_ENTRY_LEAF_ONLY
#define BLAS_STATS_ACCUM_DOBB               BLAS_STATS_ENTRY_DOBB
#define BLAS_STATS_ACCUM_DOBB_LEAF_ONLY     BLAS_STATS_ENTRY_DOBB_LEAF_ONLY
#define BLAS_STATS_ACCUM_BAD_BOX_SLOTS      BLAS_STATS_ENTRY_BAD_BOX_SLOTS
#define BLAS_STATS_ACCUM_GROUPS_DONE        13      // Finished thread groups

#define BLAS_STATS_ACCUM_FIRST_DWORD        8       // Accumulator dwords [8, 14) are zeroed by the CP
#define BLAS_STATS_ACCUM_DWORDS             6

#endif
