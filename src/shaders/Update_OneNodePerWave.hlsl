/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
// Update with one fat leaf per wave (BLAS with WideNodeMergingFlags 0x4): the first lane group of each wave refits one
// fat leaf and the other lane groups exit.
#include "Update_Common.hlsli"

//======================================================================================================================
// Main Function : Update_OneNodePerWave
//======================================================================================================================
[RootSignature(RootSig)]
[numthreads(BUILD_THREADGROUP_SIZE, 1, 1)]
void Update_OneNodePerWave(
    uint globalId : SV_DispatchThreadID,
    uint localId  : SV_GroupThreadID)
{
#if GPURT_BUILD_RTIP3_1
    const UpdateArgs args = BeginUpdate(globalId, localId);

    // The build wrote one group per fat leaf. It keeps four per wave for too many fat leaves (WriteUpdateGroupCount),
    // and a BLAS built with another mapping can have fewer groups: then take four per wave.
    uint fatLeafIndex = globalId / RTIP3_1_UPDATE_THREADS_PER_FAT_LEAF;
    if (args.numGroups >= args.numFatLeaves)
    {
        // One fat leaf per wave: the first lane group of thread group g takes fat leaf g
        const bool isFirstLaneGroup = (localId < RTIP3_1_UPDATE_THREADS_PER_FAT_LEAF);
        fatLeafIndex = isFirstLaneGroup ? (globalId / BUILD_THREADGROUP_SIZE) : INVALID_IDX;
    }

    if (fatLeafIndex < args.numFatLeaves)
    {
        RefitFromFatLeaf(fatLeafIndex, args.metadataSize, args.header);
    }

    EndUpdate(localId);
#endif
}
