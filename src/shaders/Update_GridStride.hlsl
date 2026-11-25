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
// Update with four fat leaves per wave over fewer waves (BLAS with WideNodeMergingFlags 0x8). The build dispatches
// SIMD count x UPDATE3_1_GRID_STRIDE_WAVES_PER_SIMD waves (WriteUpdateGroupCount). The fat leaf list is split into tasks
// of four fat leaves, and each wave takes the next task as soon as it finishes one (task counter, TaskMacros.hlsl), so a
// wave that finishes early does not wait for a fixed share. Any group count covers the list.
#include "Update_Common.hlsli"

//======================================================================================================================
// Main Function : Update_GridStride
//======================================================================================================================
[RootSignature(RootSig)]
[numthreads(BUILD_THREADGROUP_SIZE, 1, 1)]
void Update_GridStride(
    uint globalId : SV_DispatchThreadID,
    uint localId  : SV_GroupThreadID)
{
#if GPURT_BUILD_RTIP3_1
    const UpdateArgs args = BeginUpdate(globalId, localId);

    // Continue the task counter of the start tasks (BeginUpdate)
    uint waveId       = args.waveId;
    uint numTasksWait = args.numTasksWait;

    // One task per four fat leaves. BEGIN_TASK gives the task's globalId: lane group k of task t takes fat leaf 4t + k.
    const uint fatLeavesPerWave = BUILD_THREADGROUP_SIZE / RTIP3_1_UPDATE_THREADS_PER_FAT_LEAF;
    const uint numTasks         = RoundUpQuotient(args.numFatLeaves, fatLeavesPerWave);

    BEGIN_TASK(numTasks);
    const uint fatLeafIndex = globalId / RTIP3_1_UPDATE_THREADS_PER_FAT_LEAF;
    if (fatLeafIndex < args.numFatLeaves)
    {
        RefitFromFatLeaf(fatLeafIndex, args.metadataSize, args.header);
    }
    // Last task: no need to wait for the other waves
    END_TASK_NO_WAIT(numTasks);

    EndUpdate(localId);
#endif
}
