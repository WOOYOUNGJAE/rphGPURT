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
#ifndef _PRIMITIVE_STRUCTURE_ENCODER_3_1_HLSL
#define _PRIMITIVE_STRUCTURE_ENCODER_3_1_HLSL
#include "../shadersClean/common/PrimitiveStructure.hlsli"

// A batch is the immediate triangle children of one internal node (max one prim range * 8 children)
#define COMPRESSION_BATCH_SIZE (PRIM_COMP_MAX_RANGE_SIZE_3_1 * 8)

// Cluster scan ops are required to increase this to 2
#define COMPRESSION_BATCHES_PER_GROUP 1

//======================================================================================================================
struct CompBatchParentInfo
{
    uint boxNodeOffset;
    uint numBoxChildren;
    uint numLeafChildren;
};

//======================================================================================================================
uint PackCompBatchParentInfo(
    uint boxNodeOffset,
    uint numBoxChildren,
    uint numLeafChildren)
{
    // Pack numBoxChildren and numLeafChildren bits in the 128-byte aligned boxNodeOffset
    // numBoxChildren    :  3 // Maximum value is 7 since we only queue internal nodes with at least one leaf node
    // numLeafChildren   :  4 // Maximum valid is 8 as an BVH8 internal node can have maximum of 8 leaf children
    // currentNodeOffset : 25 // 128-byte aligned node offset. Lower 7 bits are guaranteed to be 0
    //
    const uint packedParentInfo = boxNodeOffset | (numLeafChildren << 3) | numBoxChildren;
    return packedParentInfo;
}

//======================================================================================================================
CompBatchParentInfo UnpackCompBatchParentInfo(
    uint packedParentInfo)
{
    const CompBatchParentInfo info =
    {
        packedParentInfo & ~bits(7),
        packedParentInfo & bits(3),
        (packedParentInfo >> 3) & bits(4),
    };

    return info;
}

//======================================================================================================================
uint GetLdsOffsetPrimStruct(uint baseLdsOffset)
{
    return baseLdsOffset + g_laneGroup.groupIndex * PRIMITIVE_STRUCT_SIZE_IN_DW;
}

//======================================================================================================================
uint GetLdsOffsetVertex(uint baseLdsOffset)
{
    // Vertex data follows the 128-byte primitive packet data in LDS
    return baseLdsOffset +
           (PRIMITIVE_STRUCT_SIZE_IN_DW * COMPRESSION_BATCHES_PER_GROUP) +
           (g_laneGroup.groupIndex * (COMPRESSION_BATCH_SIZE * 9));
}

//======================================================================================================================
uint GetLdsOffsetVertexLink(uint baseLdsOffset)
{
    // One link per vertex slot follows the vertex data in LDS
    return baseLdsOffset +
           (PRIMITIVE_STRUCT_SIZE_IN_DW * COMPRESSION_BATCHES_PER_GROUP) +
           (COMPRESSION_BATCH_SIZE * 9 * COMPRESSION_BATCHES_PER_GROUP) +
           (g_laneGroup.groupIndex * (COMPRESSION_BATCH_SIZE * 3));
}

namespace PrimStruct3_1 {
//======================================================================================================================
// Primitive structure encoder object. Mangages the data layout and writing of primitive structures.
struct Encoder
{
    uint   baseLdsOffset; // Offset in group shared memory where the primitive packet is constructed

    float3 prefixSource;

    uint triPairCount;

    uint payloadXLength;
    uint payloadYLength;
    uint payloadZLength;

    uint trailingZeroLength;

    uint geoIdAnchor;
    uint geoIdAnchorSize;
    uint geoIdPayloadSize;

    uint primIdAnchor;
    uint primIdAnchorSize;
    uint primIdPayloadSize;

    uint indexSectionMidpoint;

    // Important!! Fixed Layout Apply.
    Encoder ReadFromLane(uint index)
    {
        Encoder newEncoder;

        newEncoder.baseLdsOffset = g_laneGroup.Broadcast(baseLdsOffset, index);

        newEncoder.prefixSource = g_laneGroup.Broadcast(prefixSource, index);

        newEncoder.triPairCount = g_laneGroup.Broadcast(triPairCount, index);

        newEncoder.geoIdAnchor  = g_laneGroup.Broadcast(geoIdAnchor, index);
        newEncoder.primIdAnchor = g_laneGroup.Broadcast(primIdAnchor, index);

        if (Settings.primCompressionFlags & PrimCompFlags::FixedLayout)
        {
            newEncoder.payloadXLength = FIXED_PAYLOAD_X_LENGTH;
            newEncoder.payloadYLength = FIXED_PAYLOAD_Y_LENGTH;
            newEncoder.payloadZLength = FIXED_PAYLOAD_Z_LENGTH;

            newEncoder.trailingZeroLength = FIXED_TRAILING_ZERO_LENGTH;

            newEncoder.geoIdAnchorSize  = FIXED_GEOID_ANCHOR_SIZE;
            newEncoder.geoIdPayloadSize = FIXED_GEOID_PAYLOAD_SIZE;

            newEncoder.primIdAnchorSize  = FIXED_PRIMID_ANCHOR_SIZE;
            newEncoder.primIdPayloadSize = FIXED_PRIMID_PAYLOAD_SIZE;

            newEncoder.indexSectionMidpoint = FIXED_INDEX_SECTION_MIDPOINT;
        }
        else
        {
            newEncoder.payloadXLength = g_laneGroup.Broadcast(payloadXLength, index);
            newEncoder.payloadYLength = g_laneGroup.Broadcast(payloadYLength, index);
            newEncoder.payloadZLength = g_laneGroup.Broadcast(payloadZLength, index);

            newEncoder.trailingZeroLength = g_laneGroup.Broadcast(trailingZeroLength, index);

            newEncoder.geoIdAnchorSize  = g_laneGroup.Broadcast(geoIdAnchorSize, index);
            newEncoder.geoIdPayloadSize = g_laneGroup.Broadcast(geoIdPayloadSize, index);

            newEncoder.primIdAnchorSize  = g_laneGroup.Broadcast(primIdAnchorSize, index);
            newEncoder.primIdPayloadSize = g_laneGroup.Broadcast(primIdPayloadSize, index);

            newEncoder.indexSectionMidpoint = g_laneGroup.Broadcast(indexSectionMidpoint, index);
        }

        return newEncoder;
    }

    uint PrefixXLength()
    {
        return 32 - trailingZeroLength - payloadXLength;
    }

    uint PrefixYLength()
    {
        return 32 - trailingZeroLength - payloadYLength;
    }

    uint PrefixZLength()
    {
        return 32 - trailingZeroLength - payloadZLength;
    }

    uint VertexPayloadLength()
    {
        return payloadXLength + payloadYLength + payloadZLength;
    }

    uint VertexPrefixesLength()
    {
        return PrefixXLength() + PrefixYLength() + PrefixZLength();
    }

    void WriteDw(uint offset, uint data)
    {
        SharedMem[GetLdsOffsetPrimStruct(baseLdsOffset) + offset] = data;
    }

    void OrSharedMemBits(uint offset, uint data)
    {
        InterlockedOr(SharedMem[GetLdsOffsetPrimStruct(baseLdsOffset) + offset], data);
    }

    // Write value at the specified bit offset in the LDS primitive structure
    void WritePackedBits(
        uint startBitOffset, // Offset of the bit from the beginning of the structure
        uint length,         // Length of the bits to write
        uint value)          // Value to store in memory
    {
        if (length > 0)
        {
            const uint startDwOffset = startBitOffset / 32u;
            const uint shiftBitsInDw = startBitOffset % 32u;

            const uint64_t value64 = uint64_t(value) << shiftBitsInDw;

            OrSharedMemBits(startDwOffset, LowPart(value64));
            if ((length + shiftBitsInDw) >= 32)
            {
                OrSharedMemBits(startDwOffset + 1, HighPart(value64));
            }
        }
    }

    void WriteVertexPrefixes(float3 prefixes)
    {
        const uint x = (asuint(prefixes.x) >> (32 - PrefixXLength()));
        const uint y = (asuint(prefixes.y) >> (32 - PrefixYLength()));
        const uint z = (asuint(prefixes.z) >> (32 - PrefixZLength()));

        WritePackedBits(TRI_STRUCT_HEADER_SIZE_BITS, PrefixXLength(), x);
        WritePackedBits(TRI_STRUCT_HEADER_SIZE_BITS + PrefixXLength(), PrefixYLength(), y);
        WritePackedBits(TRI_STRUCT_HEADER_SIZE_BITS + PrefixXLength() + PrefixYLength(), PrefixZLength(), z);
    }

    void WritePrimIdAnchor(uint anchor)
    {
        WritePackedBits(indexSectionMidpoint, primIdAnchorSize, anchor);
    }

    void WriteGeoIdAnchor(uint anchor)
    {
        WritePackedBits((indexSectionMidpoint - geoIdAnchorSize), geoIdAnchorSize, anchor);
    }

    void WriteGeoIdPayload(uint triIndex, uint geoId)
    {
        if (triIndex > 0)
        {
            const uint startOffset = indexSectionMidpoint - geoIdAnchorSize - triIndex * geoIdPayloadSize;
            WritePackedBits(startOffset, geoIdPayloadSize, geoId & bits(geoIdPayloadSize));
        }
    }

    void WritePrimIdPayload(uint triIndex, uint primId)
    {
        if (triIndex > 0)
        {
            const uint startOffset = indexSectionMidpoint + primIdAnchorSize + (triIndex - 1) * primIdPayloadSize;
            WritePackedBits(startOffset, primIdPayloadSize, primId & bits(primIdPayloadSize));
        }
    }

    void WriteVertexComponent(const uint startBitOffset, const uint payloadLength, float f)
    {
        const uint component = asuint(f) >> trailingZeroLength;
        WritePackedBits(startBitOffset, payloadLength, component & bits(payloadLength));
    }

    void WriteVertex(uint vertexIndex, float3 vertex)
    {
        const uint sectionOffset = TRI_STRUCT_HEADER_SIZE_BITS + VertexPrefixesLength();

        if (Settings.geometryType == GEOMETRY_TYPE_AABBS)
        {
            // store full vertex for Procedural nodes
            const uint32 dwBitLength = 0x20;
            uint32_t offset = TRI_STRUCT_HEADER_SIZE_BITS + vertexIndex * dwBitLength * 3;
            WriteVertexComponent(offset, dwBitLength, vertex.x);
            offset += dwBitLength;
            WriteVertexComponent(offset, dwBitLength, vertex.y);
            offset += dwBitLength;
            WriteVertexComponent(offset, dwBitLength, vertex.z);
        }
        else
        {
            uint startBitOffset = sectionOffset + vertexIndex * VertexPayloadLength();
            WriteVertexComponent(startBitOffset, payloadXLength, vertex.x);

            startBitOffset += payloadXLength;
            WriteVertexComponent(startBitOffset, payloadYLength, vertex.y);

            startBitOffset += payloadYLength;
            WriteVertexComponent(startBitOffset, payloadZLength, vertex.z);
        }
    }

    void WriteTriPairDesc(uint triPairIndex, uint desc)
    {
        const uint startOffset = 1024 - TRI_PAIR_DESC_SIZE * (triPairIndex + 1);
        WritePackedBits(startOffset, TRI_PAIR_DESC_SIZE, desc);
    }

    void WriteMetadata()
    {
        uint headerLo;
        uint headerHi;

        if (Settings.primCompressionFlags & PrimCompFlags::FixedLayout)
        {
            headerLo = PackFixedMetadataHeaderBitsLo(triPairCount);
            headerHi = PackFixedMetadataHeaderBitsHi();
        }
        else
        {
            headerLo = PackMetadataHeaderBitsLo(
                payloadXLength - 1,
                payloadYLength - 1,
                payloadZLength - 1,
                trailingZeroLength,
                geoIdAnchorSize / 2,
                geoIdPayloadSize / 2,
                triPairCount - 1,
                false /* unorm16 */);

            headerHi = PackMetadataHeaderBitsHi(
                primIdAnchorSize,
                primIdPayloadSize,
                indexSectionMidpoint);
        }

        WriteDw(0, headerLo);
        WriteDw(1, headerHi);

        WriteVertexPrefixes(prefixSource);
        WritePrimIdAnchor(primIdAnchor);
        WriteGeoIdAnchor(geoIdAnchor);
    }
    // Important!! Fixed Layout Apply.
    bool TryInit(
        float3 prefixSource_,
        uint3  planeDiffs,
        uint   planeUnion,
        uint   primIdDiff,
        uint   geoIdDiff,
        uint   primIdAnchor_,
        uint   geoIdAnchor_,
        uint   uniqueVertexCount,
        uint   triPairCount_)
    {
        prefixSource = prefixSource_;
        primIdAnchor = primIdAnchor_;
        geoIdAnchor  = geoIdAnchor_;
        triPairCount = triPairCount_;

        const bool useFixedLayout = Settings.primCompressionFlags & PrimCompFlags::FixedLayout;

        const bool enableTrailingZeroCompression =
            (Settings.primCompressionFlags & PrimCompFlags::EnableTrailingZeroAndPrefix) && (useFixedLayout == false);

        trailingZeroLength = enableTrailingZeroCompression ? CommonTrailingZeroBits(planeUnion) : 0;

        uint3 prefixLengths = enableTrailingZeroCompression ? CommonPrefixBits(planeDiffs) : uint3(0, 0, 0);

        prefixLengths = min(prefixLengths, (31 - trailingZeroLength).xxx);

        const uint vertexSize =	96 - (prefixLengths.x + prefixLengths.y + prefixLengths.z + 3 * trailingZeroLength);

        primIdPayloadSize = useFixedLayout ? FIXED_PRIMID_PAYLOAD_SIZE : (32 - LeadingZeroBits(primIdDiff));
        primIdAnchorSize  = useFixedLayout ? FIXED_PRIMID_ANCHOR_SIZE  : (32 - LeadingZeroBits(primIdAnchor));

        // Note, geometry ID payload and anchor lengths are encoded as multiple of 2. We need to align the lengths here
        // to account for that
        geoIdPayloadSize = useFixedLayout ? FIXED_GEOID_PAYLOAD_SIZE : Pow2Align(32 - LeadingZeroBits(geoIdDiff), 2);
        geoIdAnchorSize  = useFixedLayout ? FIXED_GEOID_ANCHOR_SIZE  : Pow2Align(32 - LeadingZeroBits(geoIdAnchor), 2);

        const uint pairDescDataSize  = triPairCount * TRI_PAIR_DESC_SIZE;
        const uint idPayloadDataSize = ((2 * triPairCount) - 1) * (primIdPayloadSize + geoIdPayloadSize);
        const uint vertexDataSize    = uniqueVertexCount * vertexSize;
        const uint anchorSizes
            = prefixLengths.x + prefixLengths.y + prefixLengths.z + primIdAnchorSize + geoIdAnchorSize;
        const uint requisiteBits
            = pairDescDataSize + idPayloadDataSize + vertexDataSize + anchorSizes;

        const uint maxUniqueVertexCount = useFixedLayout ? 7 : 16;
        const uint maxTriPairCount      = useFixedLayout ? 3 : 8;

        const bool valid =
            ((requisiteBits <= TRI_STRUCT_DATA_SECTION_SIZE_BITS) &&
             (uniqueVertexCount <= maxUniqueVertexCount) &&
             (triPairCount <= maxTriPairCount));

        if (valid)
        {
            GPU_ASSERT((prefixLengths.x + trailingZeroLength) <= 31);
            GPU_ASSERT((prefixLengths.y + trailingZeroLength) <= 31);
            GPU_ASSERT((prefixLengths.z + trailingZeroLength) <= 31);

            if (useFixedLayout)
            {
                payloadXLength = FIXED_PAYLOAD_X_LENGTH;
                payloadYLength = FIXED_PAYLOAD_Y_LENGTH;
                payloadZLength = FIXED_PAYLOAD_Z_LENGTH;

                indexSectionMidpoint = FIXED_INDEX_SECTION_MIDPOINT;
            }
            else
            {
                payloadXLength = (32 - prefixLengths.x - trailingZeroLength);
                payloadYLength = (32 - prefixLengths.y - trailingZeroLength);
                payloadZLength = (32 - prefixLengths.z - trailingZeroLength);

                indexSectionMidpoint = ComputeIndexSectionMidpoint(triPairCount, primIdPayloadSize, primIdAnchorSize);
            }
        }

        // One pair should always fit
        GPU_ASSERT(valid || (triPairCount > 1));

        return valid;
    }
};

struct TriCompressState
{
    // State valid for a lane with a complete prim struct
    VertexMask vertexMask;
    Encoder encoder;

    // Per-lane/primitive state
    uint basePrimIndex;
    uint triPairIdx;
    uint pairDesc;
    uint geometryId;
    uint primId;
    uint childIdx;
    bool isTri1;
    bool completePair;
    bool primRangeStart;
    bool emitPrimStruct;
};

//======================================================================================================================
bool IsTriangle0(
    uint64_t primRangeStartMask,
    uint32_t primListIndex)
{
    // Mask off start bits above this lane's index
    const uint64_t lowerPrimStartMask = primRangeStartMask & bits64(primListIndex + 1);
    // Search the start mask from the end down to find the last start point
    const uint lastPrimStart = FIRSTBITHIGH_U64(lowerPrimStartMask);
    const uint triIndexInRange = primListIndex - lastPrimStart;

    const bool isTri0 = (triIndexInRange & 1) == 0;
    return isTri0;
}

// Important!! Fixed Layout Apply.
//======================================================================================================================
// Determine which primitives to compress together. Each primitive is assigned to a lane. On each iteration of the
// main loop, each lane attempts to compress with all active preceeding lanes. The highest successful lane determines
// a final primitive structure.
static TriCompressState ComputeCompressedRanges(
    AccelStructOffsets offsets,
    uint     baseLdsOffset,
    uint     primListIndex,
    uint     geomId,
    uint     primId,
    float3   v0,
    float3   v1,
    float3   v2,
    uint64_t primRangeStartMask,
    uint64_t primRangeEndMask,
    uint     numBoxChildren,
    bool     isOpaque,
    bool     isTri0,
    bool     compressMultiplePairs)
{
    const uint maxCompressedPairs = (Settings.primCompressionFlags & PrimCompFlags::FixedLayout) ? 3 : 8;
    const uint maxPairs = compressMultiplePairs ? maxCompressedPairs : 1;

    // Vertex search of the pass loop below (RtPrimCompressionFlags bits [4:5], see PrimCompMode). Spec constant.
    //   mode 0: every remaining lane runs FindVerts in every pass (original)
    //   mode 1: only lanes that can still fit in the structure (pair count <= maxPairs) run FindVerts
    //   mode 2: per-batch links to the previous slot with the same vertex, walked in the pass loop (lanes as in mode 1)
    const bool searchMode1 = (Settings.primCompressionFlags & PrimCompFlags::SearchMode1) != 0;
    const bool searchMode2 = (Settings.primCompressionFlags & PrimCompFlags::SearchMode2) != 0;

    const uint childIdx       = countbits64(primRangeStartMask & bits64(primListIndex)) + numBoxChildren;
    const bool primRangeStart = (primRangeStartMask & bit64(primListIndex));
    const bool primRangeEnd   = (primRangeEndMask & bit64(primListIndex));

    const bool isTri1 = (isTri0 == false);

    // This lane is the end of a pair if it's tri1 or tri0 at the end of a range
    const bool completePair = primRangeEnd || isTri1;

    uint triPairIdx = 0;

    uint primsDone = 0;

    TrianglePairDesc pairDesc;

    StoreVert(baseLdsOffset, primListIndex * 3 + 0, v0);
    StoreVert(baseLdsOffset, primListIndex * 3 + 1, v1);
    StoreVert(baseLdsOffset, primListIndex * 3 + 2, v2);

    // (Search mode 2) Link each vertex slot of this triangle to the previous slot in the batch that holds the same
    // vertex. A structure starts at a tri0 and holds at most maxPairs pairs, so only the (2 * maxPairs - 1) prims before
    // this one can share a structure with it, and only those are searched. This loop has no early exit, so its LDS reads
    // can be issued back to back.
    const uint ownSlot = primListIndex * 3;
    const uint linkLds = GetLdsOffsetVertexLink(baseLdsOffset);
    if (searchMode2)
    {
        const uint lookbackPrims = (2 * maxPairs) - 1;
        const uint lowSlot = (primListIndex > lookbackPrims) ? ((primListIndex - lookbackPrims) * 3) : 0;

        uint3 prevSlot = uint3(INVALID_IDX, INVALID_IDX, INVALID_IDX);
        for (int i = int(ownSlot) + 1; i >= int(lowSlot); i--)
        {
            const float3 vert = ReadVert(baseLdsOffset, uint(i));
            prevSlot.x = ((prevSlot.x == INVALID_IDX) && (uint(i) < ownSlot)       && all(v0 == vert)) ? uint(i) : prevSlot.x;
            prevSlot.y = ((prevSlot.y == INVALID_IDX) && (uint(i) < (ownSlot + 1)) && all(v1 == vert)) ? uint(i) : prevSlot.y;
            prevSlot.z = ((prevSlot.z == INVALID_IDX)                              && all(v2 == vert)) ? uint(i) : prevSlot.z;
        }

        SetSharedMem(linkLds + ownSlot + 0, prevSlot.x);
        SetSharedMem(linkLds + ownSlot + 1, prevSlot.y);
        SetSharedMem(linkLds + ownSlot + 2, prevSlot.z);
    }

    uint numPrimsDone  = 0;
    uint basePrimIndex = 0;

    Encoder encoder;
    encoder.baseLdsOffset = baseLdsOffset;

    VertexMask vertexMask;
    uint uniqueVertexCount = 0;

    bool emitPrimStruct = false;
    bool triDone = false;

    while (triDone == false)
    {
        basePrimIndex = numPrimsDone;

        const uint3 prefixSource = g_laneGroup.ReadFirstActiveLane(asuint(v0));

        const uint primIdAnchor = g_laneGroup.ReadFirstActiveLane(primId);
        const uint geomIdAnchor = g_laneGroup.ReadFirstActiveLane(geomId);

        if (g_laneGroup.IsFirstActiveLane())
        {
            GPU_ASSERT(numPrimsDone == g_laneGroup.laneIndex);
        }

        // Pair count of this lane if the structure starts at basePrimIndex. It only depends on the active lanes.
        const uint triPairDescCount = g_laneGroup.ExclusivePrefixSumBool(isTri0) + isTri0;

        // Index of each vertex = its first slot at or after the structure base.
        // A lane past maxPairs cannot be part of this structure (TryInit is not called for it) and its vertex mask bits
        // only reach lanes above it, so search modes 1 and 2 skip its search; the result is unchanged.
        // FindVerts scans [basePrimIndex, primListIndex] serially and the wave waits for its slowest lane, so in mode 0 a
        // pass costs as much as the distance from the base to the last prim of the batch.
        uint3 index = uint3(0, 0, 0);
        if (searchMode2 == false)
        {
            if ((searchMode1 == false) || (triPairDescCount <= maxPairs))
            {
                index = FindVerts(baseLdsOffset, basePrimIndex, primListIndex, v0, v1, v2);
            }
        }
        else if (triPairDescCount <= maxPairs)
        {
            // (Search mode 2) Walk the links back while they stay at or after the base.
            const uint baseSlot = basePrimIndex * 3;

            [unroll]
            for (uint c = 0; c < 3; c++)
            {
                uint slot = ownSlot + c;
                uint prev = GetSharedMem(linkLds + slot);
                while ((prev != INVALID_IDX) && (prev >= baseSlot))
                {
                    slot = prev;
                    prev = GetSharedMem(linkLds + slot);
                }
                index[c] = slot - baseSlot;
            }
        }

        uint3 planeDiffs =
            (asuint(v0) ^ prefixSource) |
            (asuint(v1) ^ prefixSource) |
            (asuint(v2) ^ prefixSource);

        planeDiffs = g_laneGroup.PostfixOr(planeDiffs);

        uint planeUnion =
            asuint(v0.x) | asuint(v0.y) | asuint(v0.z) |
            asuint(v1.x) | asuint(v1.y) | asuint(v1.z) |
            asuint(v2.x) | asuint(v2.y) | asuint(v2.z);

        planeUnion = g_laneGroup.PostfixOr(planeUnion);

        uint geoIdDiff  = g_laneGroup.PostfixOr(geomIdAnchor ^ geomId);
        uint primIdDiff = g_laneGroup.PostfixOr(primIdAnchor ^ primId);

        vertexMask.Init();
        vertexMask.SetBit(index[0]);
        vertexMask.SetBit(index[1]);
        vertexMask.SetBit(index[2]);
        vertexMask.PostfixOr();

        uniqueVertexCount = vertexMask.CountBits();

        bool valid = false;

        if ((triPairDescCount <= maxPairs) && completePair)
        {
            valid = encoder.TryInit(asfloat(prefixSource),
                                    planeDiffs,
                                    planeUnion,
                                    primIdDiff,
                                    geoIdDiff,
                                    primIdAnchor,
                                    geomIdAnchor,
                                    uniqueVertexCount,
                                    triPairDescCount);
        }

        const uint64_t validMask = g_laneGroup.Ballot(valid);
        GPU_ASSERT(validMask != 0);
        const uint lastValidTri = FIRSTBITHIGH_U64(validMask);

        if (primListIndex <= lastValidTri)
        {
            emitPrimStruct = (primListIndex == lastValidTri);

            uint3 vertIndicesInStruct = vertexMask.CompactIndices(index);

            pairDesc.Init();
            pairDesc.InitTri0(vertIndicesInStruct, isOpaque, primId, geomId);

            const uint prevLanePairDesc = g_laneGroup.Broadcast(pairDesc.triDesc, g_laneGroup.laneIndex - 1);

            if (completePair)
            {
                if (isTri1)
                {
                    // This lane is tri1, shift the tri0 data into the tri1 position
                    pairDesc.triDesc >>= (TRI_DESC_TRI0_DOUBLE_SIDED_SHIFT - 1);
                    // Or in the tri0 data from the previous lane
                    pairDesc.triDesc |= prevLanePairDesc;
                }

                pairDesc.SetPrimRangeStopBit(primRangeEnd);
            }

            triPairIdx = triPairDescCount - 1;

            triDone = true;
        }

        numPrimsDone = lastValidTri + 1;
    }

    TriCompressState outputState;

    outputState.encoder = encoder;
    outputState.vertexMask = vertexMask;
    outputState.basePrimIndex = basePrimIndex;
    outputState.triPairIdx = triPairIdx;
    outputState.pairDesc = pairDesc.GetData();
    outputState.geometryId = geomId;
    outputState.primId = primId;
    outputState.childIdx = childIdx;
    outputState.isTri1 = isTri1;
    outputState.completePair = completePair;
    outputState.primRangeStart = primRangeStart;
    outputState.emitPrimStruct = emitPrimStruct;

    return outputState;
}

//======================================================================================================================
static void EmitPrimitiveStructures(
    AccelStructOffsets offsets,
    uint baseLdsOffset,
    TriCompressState triState,
    uint parentBoxOffset,
    uint numBoxChildren,
    uint64_t primRangeEndMask,
    uint64_t emitPrimStructMask,
    uint primStructIdx,
    uint primStructOffset)
{
    const uint primStructCount = countbits64(emitPrimStructMask);

    const uint primListIndex = g_laneGroup.laneIndex;

    if (primListIndex < primStructCount)
    {
        // Write prim struct offsets
        const uint lanePrimStructIdx = g_laneGroup.ReadFirstLane(primStructIdx) + primListIndex;
        const uint lanePrimStructOffset = primStructOffset + PRIMITIVE_STRUCT_SIZE_IN_BYTE * primListIndex;
        DstBuffer.Store(offsets.primNodePtrs + (lanePrimStructIdx * sizeof(uint)), lanePrimStructOffset);
    }

    // For each prim struct, each lane writes to LDS, then data is copied to memory
    uint firstPrimListIndex = 0;
    while (emitPrimStructMask != 0)
    {
        const uint emitIndex = FIRSTBITLOW_U64(emitPrimStructMask);

        for (uint i = g_laneGroup.laneIndex; i < PRIMITIVE_STRUCT_SIZE_IN_DW; i += g_laneGroup.groupSize)
        {
            SharedMem[GetLdsOffsetPrimStruct(baseLdsOffset) + i] = 0;
        }

        GroupMemoryBarrierWithGroupSync();

        Encoder emitEncoder = triState.encoder.ReadFromLane(emitIndex);

        if (g_laneGroup.IsFirstActiveLane())
        {
            emitEncoder.WriteMetadata();
        }

        GroupMemoryBarrierWithGroupSync();

        VertexMask emitVertexMask = triState.vertexMask.ReadFromLane(emitIndex);

        const uint basePrimIndex = g_laneGroup.Broadcast(triState.basePrimIndex, emitIndex);

        for (uint vert = g_laneGroup.laneIndex; vert < COMPRESSION_BATCH_SIZE * 3; vert += g_laneGroup.groupSize)
        {
            if (emitVertexMask.TestBit(vert))
            {
                const uint vertIndexInStruct = emitVertexMask.CompactIndex(vert);
                const uint absoluteVertexIndex = vert + basePrimIndex * 3;
                emitEncoder.WriteVertex(vertIndexInStruct, ReadVert(baseLdsOffset, absoluteVertexIndex));
            }
        }

        if ((primListIndex >= firstPrimListIndex) && (primListIndex <= emitIndex))
        {
            if (triState.completePair)
            {
                emitEncoder.WriteTriPairDesc(triState.triPairIdx, triState.pairDesc);
            }

            const uint triIdx = (triState.triPairIdx * 2) + triState.isTri1;

            emitEncoder.WriteGeoIdPayload(triIdx, triState.geometryId);
            emitEncoder.WritePrimIdPayload(triIdx, triState.primId);

            const uint nodeType = (triState.triPairIdx >= 4) ? (triState.triPairIdx + 4) : triState.triPairIdx;

            if (triState.primRangeStart)
            {
                // Find the first range end bit at or above the current position
                const uint rangeEnd = FIRSTBITLOW_U64(primRangeEndMask & ~bits64(primListIndex));
                // Mask of primitives in the range
                const uint64_t rangeMask = bits64(rangeEnd + 1) - bits64(primListIndex);
                // Count the packets that will be emitted in the range
                const uint rangePacketCount = countbits64(rangeMask & emitPrimStructMask);
                GPU_ASSERT(rangePacketCount < 16);

                // Update child type in the box node
                const uint childInfoOffset =
                    parentBoxOffset +
                    QUANTIZED_BVH8_NODE_OFFSET_CHILD_INFO_0 +
                    (triState.childIdx * QUANTIZED_NODE_CHILD_INFO_STRIDE) +
                    QUANTIZED_NODE_CHILD_INFO_OFFSET_MAXY_MAXZ_NODE_TYPE_AND_RANGE;

                DstBuffer.InterlockedOr(childInfoOffset, (nodeType << 24) | (rangePacketCount << 28));
            }
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint i = g_laneGroup.laneIndex; i < PRIMITIVE_STRUCT_SIZE_IN_DW; i += g_laneGroup.groupSize)
        {
            DstBuffer.Store(primStructOffset + i * 4,
                            SharedMem[GetLdsOffsetPrimStruct(baseLdsOffset) + i]);
        }

        primStructOffset += PRIMITIVE_STRUCT_SIZE_IN_BYTE;

        emitPrimStructMask &= emitPrimStructMask - 1;
        firstPrimListIndex = emitIndex + 1;
    }
}

//=====================================================================================================================
// Create a Primitive Structure containing a single triangle pair
// NOTE: This function is used by the Trivial Builder, so it cannot use scratch memory or LDS.
static void WritePairPrimStructVariableLayout(
    TriangleData tri0,
    bool         tri0Opaque,
    uint         tri0PrimitiveIndex,
    uint         tri0GeometryIndex,
    bool         isTri1Valid,
    TriangleData tri1,
    bool         tri1Opaque,
    uint         tri1PrimitiveIndex,
    uint         tri1GeometryIndex,
    uint         nodeOffset)
{
    uint d[PRIMITIVE_STRUCT_SIZE_IN_DW];

    for (uint i = 0; i < PRIMITIVE_STRUCT_SIZE_IN_DW; ++i)
    {
        d[i] = 0;
    }

    const uint indexSectionMidpoint = ComputeIndexSectionMidpoint(1, 31, 31);

    if (Settings.geometryType == GEOMETRY_TYPE_AABBS)
    {
        // HW would detect "malformed_vtx" based on index and payloadlength
        // Since procedural node index are 0xF
        // here we manually pack payloadLength to be 1 (payloadLengthMinusOne = 0)
        // in order to prevent HW decode proceduralNodes as "malformed_vtx"
        //
        // i.e., procedural node needs to be packed as:
        // payloadXLengthMinusOne = 0;      // decode as 1
        // payloadYLengthMinusOne = 0;      // decode as 1
        // payloadZLengthMinusOne = 0;      // decode as 1
        // trailingZeroLength     = 0;

        d[0] = PackMetadataHeaderBitsLo(0, 0, 0, 0, 15, 15, 0, false);
        d[1] = PackMetadataHeaderBitsHi(31, 31, indexSectionMidpoint);

        // tri0 v0 is encoded as float3(0, 0, 0) to report invalid triangle vertices as origin.

        // AABB min/max are written to unused section of the primitive packet.
        d[5]  = asuint(tri0.v0.x);
        d[6]  = asuint(tri0.v0.y);
        d[7]  = asuint(tri0.v0.z);
        d[8]  = asuint(tri0.v1.x);
        d[9]  = asuint(tri0.v1.y);
        d[10] = asuint(tri0.v1.z);

        // Geometry IDs
        d[28] = bitFieldInsert(d[28], 7, 25, tri0GeometryIndex);
        d[29] = bitFieldInsert(0,     0,  5, tri0GeometryIndex >> 25);

        // Index section midpoint

        // Primitive IDs
        d[29] = bitFieldInsert(d[29], 5, 27, tri0PrimitiveIndex);
        d[30] = bitFieldInsert(d[30], 0,  4, tri0PrimitiveIndex >> 27);

        // Triangle descriptor
        uint triDesc = 1 << TRI_DESC_PRIM_RANGE_STOP_BIT_SHIFT;
        // Mark this node as procedural by setting triangle 0 indices to 0xff, 0xff, x
        PackedTriDescSetTriangle(0, 0xff, 0xff, 0xff, (uint) tri0Opaque, triDesc);
        // Encode triangle 1 as degenerate triangle at vertex0.
        PackedTriDescSetTriangle(1, 0, 0, 0, 0, triDesc);

        d[31] = bitFieldInsert(d[31], 3, 29, triDesc);
    }
    else
    {
        // Store full vertex with no compression
        //
        // Each vertex is stored as 3 DW (32bit) without prefix
        //
        // Header ----------- 52 bits
        // Vertex Data ------ 576 bits (96 * 3 * 2)
        // ----- Unused ----- 245 bits
        // geometryId1Payload 30 bits
        // geometryIdAnchor-- 30 bits
        //------------------- indexSectionMidpoint
        // primIdAnchor ----- 31 bits
        // primId1Payload---- 31 bits
        // triDesc----------- 29 bits
        //------------------- 1024 bits
        d[0] = PackMetadataHeaderBitsLo(31, 31, 31, 0, 15, 15, 0, false);

        d[1] = PackMetadataHeaderBitsHi(31, 31, indexSectionMidpoint);

        uint triDesc = 1 << TRI_DESC_PRIM_RANGE_STOP_BIT_SHIFT;

        // tri0 v0
        d[1] = bitFieldInsert(d[1], 20, 12, asuint(tri0.v0.x));
        d[2] = bitFieldInsert(0,     0, 20, asuint(tri0.v0.x) >> 12);
        d[2] = bitFieldInsert(d[2], 20, 12, asuint(tri0.v0.y));
        d[3] = bitFieldInsert(0,     0, 20, asuint(tri0.v0.y) >> 12);
        d[3] = bitFieldInsert(d[3], 20, 12, asuint(tri0.v0.z));
        d[4] = bitFieldInsert(0,     0, 20, asuint(tri0.v0.z) >> 12);
        // tri0 v1
        d[4] = bitFieldInsert(d[4], 20, 12, asuint(tri0.v1.x));
        d[5] = bitFieldInsert(0,     0, 20, asuint(tri0.v1.x) >> 12);
        d[5] = bitFieldInsert(d[5], 20, 12, asuint(tri0.v1.y));
        d[6] = bitFieldInsert(0,     0, 20, asuint(tri0.v1.y) >> 12);
        d[6] = bitFieldInsert(d[6], 20, 12, asuint(tri0.v1.z));
        d[7] = bitFieldInsert(0,     0, 20, asuint(tri0.v1.z) >> 12);

        // tri0 v2
        d[7]  = bitFieldInsert(d[7], 20, 12, asuint(tri0.v2.x));
        d[8]  = bitFieldInsert(0,     0, 20, asuint(tri0.v2.x) >> 12);
        d[8]  = bitFieldInsert(d[8], 20, 12, asuint(tri0.v2.y));
        d[9]  = bitFieldInsert(0,     0, 20, asuint(tri0.v2.y) >> 12);
        d[9]  = bitFieldInsert(d[9], 20, 12, asuint(tri0.v2.z));
        d[10] = bitFieldInsert(0,     0, 20, asuint(tri0.v2.z) >> 12);

        PackedTriDescSetTriangle(0, 0, 1, 2, (uint) tri0Opaque, triDesc);

        if (isTri1Valid)
        {
            // tri1 v0
            d[10] = bitFieldInsert(d[10], 20, 12, asuint(tri1.v0.x));
            d[11] = bitFieldInsert(0,      0, 20, asuint(tri1.v0.x) >> 12);
            d[11] = bitFieldInsert(d[11], 20, 12, asuint(tri1.v0.y));
            d[12] = bitFieldInsert(0,      0, 20, asuint(tri1.v0.y) >> 12);
            d[12] = bitFieldInsert(d[12], 20, 12, asuint(tri1.v0.z));
            d[13] = bitFieldInsert(0,      0, 20, asuint(tri1.v0.z) >> 12);
            // tri1 v1
            d[13] = bitFieldInsert(d[13], 20, 12, asuint(tri1.v1.x));
            d[14] = bitFieldInsert(0,      0, 20, asuint(tri1.v1.x) >> 12);
            d[14] = bitFieldInsert(d[14], 20, 12, asuint(tri1.v1.y));
            d[15] = bitFieldInsert(0,      0, 20, asuint(tri1.v1.y) >> 12);
            d[15] = bitFieldInsert(d[15], 20, 12, asuint(tri1.v1.z));
            d[16] = bitFieldInsert(0,      0, 20, asuint(tri1.v1.z) >> 12);
            // tri1 v2
            d[16] = bitFieldInsert(d[16], 20, 12, asuint(tri1.v2.x));
            d[17] = bitFieldInsert(0,      0, 20, asuint(tri1.v2.x) >> 12);
            d[17] = bitFieldInsert(d[17], 20, 12, asuint(tri1.v2.y));
            d[18] = bitFieldInsert(0,      0, 20, asuint(tri1.v2.y) >> 12);
            d[18] = bitFieldInsert(d[18], 20, 12, asuint(tri1.v2.z));
            d[19] = bitFieldInsert(0,      0, 20, asuint(tri1.v2.z) >> 12);

            PackedTriDescSetTriangle(1, 3, 4, 5, (uint) tri1Opaque, triDesc);

            // Geometry IDs
            d[27] = bitFieldInsert(0, 9, 23, tri1GeometryIndex);
            d[28] = bitFieldInsert(0, 0,  7, tri1GeometryIndex >> 23);

            // Primitive IDs
            d[30] = bitFieldInsert(0, 4, 28, tri1PrimitiveIndex);
            d[31] = bitFieldInsert(0, 0,  3, tri1PrimitiveIndex >> 28);
        }
        else
        {
            // Set tri1 to invalid
            PackedTriDescSetInvalidTriangle(1, triDesc);
        }

        // Geometry IDs
        d[28] = bitFieldInsert(d[28], 7, 25, tri0GeometryIndex);
        d[29] = bitFieldInsert(0,     0,  5, tri0GeometryIndex >> 25);

        // Index section midpoint

        // Primitive IDs
        d[29] = bitFieldInsert(d[29], 5, 27, tri0PrimitiveIndex);
        d[30] = bitFieldInsert(d[30], 0,  4, tri0PrimitiveIndex >> 27);

        // Triangle descriptor
        d[31] = bitFieldInsert(d[31], 3, 29, triDesc);
    }

    [unroll]
    for (uint i = 0; i < PRIMITIVE_STRUCT_SIZE_IN_DW; i += 4)
    {
        DstBuffer.Store4(nodeOffset + (i*4), uint4(d[i], d[i+1], d[i+2], d[i+3]));
    }
}

//======================================================================================================================
// Important!! Fixed Layout Apply.
// Create a Primitive Structure containing a single triangle pair
// NOTE: This function is used by the Trivial Builder, so it cannot use scratch memory or LDS.
static void WritePairPrimStructFixedLayout(
    TriangleData tri0,
    bool         tri0Opaque,
    uint         tri0PrimitiveIndex,
    uint         tri0GeometryIndex,
    bool         isTri1Valid,
    TriangleData tri1,
    bool         tri1Opaque,
    uint         tri1PrimitiveIndex,
    uint         tri1GeometryIndex,
    uint         nodeOffset)
{
    // Store full vertex with no compression
    //
    // Each vertex is stored as 3 DW (32bit) without prefix
    //
    //-------------------
    // Header                    - 52  bits
    // Vertex Data               - 672 bits (96 * 7)
    // Geometry Index 5 Payload  - 12  bits
    // Geometry Index 4 Payload  - 12  bits
    // Geometry Index 3 Payload  - 12  bits
    // Geometry Index 2 Payload  - 12  bits
    // Geometry Index 1 Payload  - 12  bits
    // Geometry Index 0 Anchor   - 24  bits
    // Primitive Index 0 Anchor  - 29  bits
    // Primitive Index 1 Payload - 18  bits
    // Primitive Index 2 Payload - 18  bits
    // Primitive Index 3 Payload - 18  bits
    // Primitive Index 4 Payload - 18  bits
    // Primitive Index 5 Payload - 18  bits
    // Padding                   - 10  bits
    // Triangle Pair Descriptor2 - 29  bits
    // Triangle Pair Descriptor1 - 29  bits
    // Triangle Pair Descriptor0 - 29  bits
    //------------------- 1024 bits total

    uint d[PRIMITIVE_STRUCT_SIZE_IN_DW];

    // Metadata header
    d[0] = PackFixedMetadataHeaderBitsLo(1);
    d[1] = PackFixedMetadataHeaderBitsHi();

    uint triDesc = 1 << TRI_DESC_PRIM_RANGE_STOP_BIT_SHIFT;

    // tri0 v0
    d[1] = bitFieldInsert(d[1], 20, 12, asuint(tri0.v0.x));
    d[2] = bitFieldInsert(0,     0, 20, asuint(tri0.v0.x) >> 12);
    d[2] = bitFieldInsert(d[2], 20, 12, asuint(tri0.v0.y));
    d[3] = bitFieldInsert(0,     0, 20, asuint(tri0.v0.y) >> 12);
    d[3] = bitFieldInsert(d[3], 20, 12, asuint(tri0.v0.z));
    d[4] = bitFieldInsert(0,     0, 20, asuint(tri0.v0.z) >> 12);
    // tri0 v1
    d[4] = bitFieldInsert(d[4], 20, 12, asuint(tri0.v1.x));
    d[5] = bitFieldInsert(0,     0, 20, asuint(tri0.v1.x) >> 12);
    d[5] = bitFieldInsert(d[5], 20, 12, asuint(tri0.v1.y));
    d[6] = bitFieldInsert(0,     0, 20, asuint(tri0.v1.y) >> 12);
    d[6] = bitFieldInsert(d[6], 20, 12, asuint(tri0.v1.z));
    d[7] = bitFieldInsert(0,     0, 20, asuint(tri0.v1.z) >> 12);

    if (Settings.geometryType == GEOMETRY_TYPE_TRIANGLES)
    {
        // tri0 v2
        d[7]  = bitFieldInsert(d[7], 20, 12, asuint(tri0.v2.x));
        d[8]  = bitFieldInsert(0,     0, 20, asuint(tri0.v2.x) >> 12);
        d[8]  = bitFieldInsert(d[8], 20, 12, asuint(tri0.v2.y));
        d[9]  = bitFieldInsert(0,     0, 20, asuint(tri0.v2.y) >> 12);
        d[9]  = bitFieldInsert(d[9], 20, 12, asuint(tri0.v2.z));
        d[10] = bitFieldInsert(0,     0, 20, asuint(tri0.v2.z) >> 12);

        if (any(tri0.v0 != 0.xxx) || any(tri0.v1 != 0.xxx) || any(tri0.v2 != 0.xxx))
        {
            PackedTriDescSetTriangle(0, 0, 1, 2, (uint) tri0Opaque, triDesc);
        }
    }
    else
    {
        d[8]  = 0;
        d[9]  = 0;
        d[10] = 0;

        PackedTriDescSetTriangle(0, 0xff, 0xff, 0xff, (uint) tri0Opaque, triDesc);
    }

    if (isTri1Valid)
    {
        // tri1 v0
        d[10] = bitFieldInsert(d[10], 20, 12, asuint(tri1.v0.x));
        d[11] = bitFieldInsert(0,      0, 20, asuint(tri1.v0.x) >> 12);
        d[11] = bitFieldInsert(d[11], 20, 12, asuint(tri1.v0.y));
        d[12] = bitFieldInsert(0,      0, 20, asuint(tri1.v0.y) >> 12);
        d[12] = bitFieldInsert(d[12], 20, 12, asuint(tri1.v0.z));
        d[13] = bitFieldInsert(0,      0, 20, asuint(tri1.v0.z) >> 12);
        // tri1 v1
        d[13] = bitFieldInsert(d[13], 20, 12, asuint(tri1.v1.x));
        d[14] = bitFieldInsert(0,      0, 20, asuint(tri1.v1.x) >> 12);
        d[14] = bitFieldInsert(d[14], 20, 12, asuint(tri1.v1.y));
        d[15] = bitFieldInsert(0,      0, 20, asuint(tri1.v1.y) >> 12);
        d[15] = bitFieldInsert(d[15], 20, 12, asuint(tri1.v1.z));
        d[16] = bitFieldInsert(0,      0, 20, asuint(tri1.v1.z) >> 12);
        // tri1 v2
        d[16] = bitFieldInsert(d[16], 20, 12, asuint(tri1.v2.x));
        d[17] = bitFieldInsert(0,      0, 20, asuint(tri1.v2.x) >> 12);
        d[17] = bitFieldInsert(d[17], 20, 12, asuint(tri1.v2.y));
        d[18] = bitFieldInsert(0,      0, 20, asuint(tri1.v2.y) >> 12);
        d[18] = bitFieldInsert(d[18], 20, 12, asuint(tri1.v2.z));
        d[19] = bitFieldInsert(0,      0, 20, asuint(tri1.v2.z) >> 12);

        if (any(tri1.v0 != 0.xxx) || any(tri1.v1 != 0.xxx) || any(tri1.v2 != 0.xxx))
        {
            PackedTriDescSetTriangle(1, 3, 4, 5, (uint) tri1Opaque, triDesc);
        }

    }
    else
    {
        d[11] = 0;
        d[12] = 0;
        d[13] = 0;
        d[14] = 0;
        d[15] = 0;
        d[16] = 0;
        d[17] = 0;
        d[18] = 0;
        d[19] = 0;

        // Set tri1 to invalid
        PackedTriDescSetInvalidTriangle(1, triDesc);
    }

    d[20] = 0;
    d[21] = 0;
    d[22] = 0;
    // Geometry Index 5 Payload
    d[22] = bitFieldInsert(d[22], 20, 12, 0/*tri5GeometryIndex*/);
    // Geometry Index 4 Payload
    d[23] = bitFieldInsert(0, 0, 12, 0);
    // Geometry Index 3 Payload
    d[23] = bitFieldInsert(d[23], 12, 12, 0);
    // Geometry Index 2 Payload
    d[23] = bitFieldInsert(d[23], 24, 8, 0);
    d[24] = bitFieldInsert(0, 0, 4, 0/*tri2GeometryIndex >> 8*/);
    // Geometry Index 1 Payload
    if (isTri1Valid) {
        d[24] = bitFieldInsert(d[24], 4, 12, tri1GeometryIndex);
    }
    // Geometry Index 0 Payload (Geo Anchor)
    d[24] = bitFieldInsert(d[24], 16, 16, tri0GeometryIndex);
    d[25] = bitFieldInsert(0, 0, 8, tri0GeometryIndex >> 16);
    // Primitive Index 0 Payload (Prim Anchor)
    d[25] = bitFieldInsert(d[25], 8, 24, tri0PrimitiveIndex);
    d[26] = bitFieldInsert(0, 0, 5, tri0PrimitiveIndex >> 24);
    // Primitive Index 1 Payload
    if (isTri1Valid) {
        d[26] = bitFieldInsert(d[26], 5, 18, tri1PrimitiveIndex);
    }
    // Primitive Index 2 Payload
    d[26] = bitFieldInsert(d[26], 23, 9, 0);
    d[27] = bitFieldInsert(0, 0, 9, 0);
    // Primitive Index 3 Payload
    d[27] = bitFieldInsert(d[27], 9, 18, 0);
    // Primitive Index 4 Payload
    d[27] = bitFieldInsert(d[27], 27, 5, 0);
    d[28] = bitFieldInsert(0, 0, 13, 0);
    //Primitive Index 5 Payload
    d[28] = bitFieldInsert(d[28], 13, 18, 0);
    // Padding
    d[28] = bitFieldInsert(d[28], 31, 1, 0);
    d[29] = bitFieldInsert(d[29], 0, 9, 0);
    // Triangle descriptor 2
    d[29] = bitFieldInsert(d[29], 9, 23, 0);
    d[30] = bitFieldInsert(0, 0, 6, 0);
    // Triangle descriptor 1
    d[30] = bitFieldInsert(d[30], 6, 26, 0);
    d[31] = bitFieldInsert(0, 0, 3, 0);
    // Triangle descriptor 0
    d[31] = bitFieldInsert(d[31], 3, 29, triDesc);

    [unroll]
    for (uint i = 0; i < PRIMITIVE_STRUCT_SIZE_IN_DW; i += 4)
    {
        DstBuffer.Store4(nodeOffset + (i*4), uint4(d[i], d[i+1], d[i+2], d[i+3]));
    }
}

//======================================================================================================================
// Create a Primitive Structure containing a single triangle pair
// NOTE: This function is used by the Trivial Builder, so it cannot use scratch memory or LDS.
static void WritePairPrimStruct(
    TriangleData tri0,
    bool         tri0Opaque,
    uint         tri0PrimitiveIndex,
    uint         tri0GeometryIndex,
    bool         isTri1Valid,
    TriangleData tri1,
    bool         tri1Opaque,
    uint         tri1PrimitiveIndex,
    uint         tri1GeometryIndex,
    uint         nodeOffset)
{
    if (Settings.primCompressionFlags & PrimCompFlags::FixedLayout)
    {
        WritePairPrimStructFixedLayout(tri0,
                                       tri0Opaque,
                                       tri0PrimitiveIndex,
                                       tri0GeometryIndex,
                                       isTri1Valid,
                                       tri1,
                                       tri1Opaque,
                                       tri1PrimitiveIndex,
                                       tri1GeometryIndex,
                                       nodeOffset);
    }
    else
    {
        WritePairPrimStructVariableLayout(tri0,
                                          tri0Opaque,
                                          tri0PrimitiveIndex,
                                          tri0GeometryIndex,
                                          isTri1Valid,
                                          tri1,
                                          tri1Opaque,
                                          tri1PrimitiveIndex,
                                          tri1GeometryIndex,
                                          nodeOffset);
    }
}

} // namespace PrimStruct3_1
#endif
