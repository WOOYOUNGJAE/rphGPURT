/*
 ***********************************************************************************************************************
 *
 *  Per-BLAS node statistics log (DeviceSettings::enableBlasStatsLog). See gpurtBlasStatsLog.h.
 *
 **********************************************************************************************************************/
#include "gpurtBlasStatsLog.h"
#include "gpurtInternal.h"
#include "shared/gpurtBlasStats.h"

#include "palSysUtil.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <vector>

namespace GpuRt
{

static constexpr uint32 RingEntries   = BLAS_STATS_RING_ENTRIES;
static constexpr uint32 MarkerBytes   = RingEntries * sizeof(uint32);
static constexpr uint32 RingSizeBytes = MarkerBytes + (RingEntries * BLAS_STATS_ENTRY_SIZE);

static_assert(BLAS_STATS_ENTRY_DWORDS * sizeof(uint32) == BLAS_STATS_ENTRY_SIZE, "BLAS stats entry size mismatch");
static_assert((MarkerBytes % BLAS_STATS_ENTRY_SIZE) == 0, "BLAS stats entries must stay 64-byte aligned");

// =====================================================================================================================
// Formats the current local time as "YYYY-MM-DD HH:MM:SS.mmm"
static void FormatNow(
    char*  pBuffer,
    size_t bufferSize)
{
    const auto   now  = std::chrono::system_clock::now();
    const time_t secs = std::chrono::system_clock::to_time_t(now);
    const uint32 ms   = static_cast<uint32>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);

    struct tm localTime = {};
#if defined(_WIN32)
    localtime_s(&localTime, &secs);
#else
    localtime_r(&secs, &localTime);
#endif

    char dateTime[32] = {};
    strftime(dateTime, sizeof(dateTime), "%Y-%m-%d %H:%M:%S", &localTime);
    snprintf(pBuffer, bufferSize, "%s.%03u", dateTime, ms);
}

// =====================================================================================================================
// Writes the process arguments after argv[0] (Linux: /proc/self/cmdline), space separated, so each header block of an
// appended log names its run (e.g. the scene). Writes an empty string where they are unavailable.
static void FormatArguments(
    char*  pBuffer,
    size_t bufferSize)
{
    pBuffer[0] = '\0';

#if defined(__linux__)
    FILE* pCmdline = fopen("/proc/self/cmdline", "rb");

    if (pCmdline != nullptr)
    {
        size_t length  = 0;
        bool   inArgv0 = true;
        int    c       = 0;

        while (((c = fgetc(pCmdline)) != EOF) && ((length + 1) < bufferSize))
        {
            if (c == '\0')
            {
                if ((inArgv0 == false) && (length > 0))
                {
                    pBuffer[length++] = ' ';
                }
                inArgv0 = false;
            }
            else if (inArgv0 == false)
            {
                pBuffer[length++] = ((c == '\n') || (c == '\r')) ? ' ' : static_cast<char>(c);
            }
        }

        // Drop the separator written for the final argument terminator.
        if ((length > 0) && (pBuffer[length - 1] == ' '))
        {
            length--;
        }
        pBuffer[length] = '\0';

        fclose(pCmdline);
    }
#endif
}

#if GPURT_BUILD_RTIP3_1
// =====================================================================================================================
// Returns the build shader primitive compression flags (PrimCompFlags) of a BLAS build, i.e. the
// BvhBuilder::m_buildConfig.primCompressionFlags it ran with. Must follow BvhBuilder: GetPrimCompressionFlags() turns
// the PrimCompMode setting into flags; AllowUpdate builds drop MultiPrim and EnableTrailingZeroAndPrefix; procedural
// BLASes keep only FixedLayout.
static uint32 GetEffectivePrimCompFlags(
    uint32 primCompMode,  // DeviceSettings::primCompressionFlags (RtPrimCompressionFlags)
    uint32 buildFlags,    // AccelStructBuildFlags of the build
    uint32 geometryType)  // GeometryType of the BLAS
{
    uint32 flags = 0;

    const uint32 layout     = primCompMode & PrimCompMode::LayoutMask;
    const uint32 searchMode = primCompMode & PrimCompMode::SearchModeMask;

    if (layout != PrimCompMode::Off)
    {
        flags = PrimCompFlags::MultiPrim | PrimCompFlags::EnableTrailingZeroAndPrefix;

        if (layout != PrimCompMode::VariableLayout)
        {
            flags |= PrimCompFlags::FixedLayout;
        }

        if (searchMode == PrimCompMode::SearchMode1)
        {
            flags |= PrimCompFlags::SearchMode1;
        }
        else if (searchMode != PrimCompMode::SearchMode0)
        {
            flags |= PrimCompFlags::SearchMode2;
        }
    }

    if (geometryType == static_cast<uint32>(GeometryType::Aabbs))
    {
        flags &= PrimCompFlags::FixedLayout;
    }
    else if (Util::TestAnyFlagSet(buildFlags, AccelStructBuildFlagAllowUpdate))
    {
        flags &= ~(PrimCompFlags::MultiPrim | PrimCompFlags::EnableTrailingZeroAndPrefix);
    }

    return flags;
}
#endif

// =====================================================================================================================
BlasStatsLog::BlasStatsLog(
    Internal::Device* pDevice)
    :
    m_pDevice(pDevice),
    m_pGpuMem(nullptr),
    m_pAccumMem(nullptr),
    m_pRing(nullptr),
    m_pFile(nullptr),
    m_pid(0),
    m_thread(),
    m_threadStarted(false),
    m_exiting(false),
    m_lastSeq(0)
{
}

// =====================================================================================================================
BlasStatsLog::~BlasStatsLog()
{
    if (m_threadStarted)
    {
        // The thread drains the ring once more before it exits. The client has idled the device by now.
        m_exiting = true;
        m_thread.Join();
    }

    if (m_pFile != nullptr)
    {
        fclose(m_pFile);
        m_pFile = nullptr;
    }

    m_pRing = nullptr;
    DestroyGpuMemory(&m_pGpuMem);
    DestroyGpuMemory(&m_pAccumMem);
}

// =====================================================================================================================
// Creates GPU memory that stays resident on the device for its whole lifetime
Pal::Result BlasStatsLog::CreateGpuMemory(
    const Pal::GpuMemoryCreateInfo& createInfo,
    Pal::IGpuMemory**               ppGpuMem)
{
    Pal::IDevice* pPalDevice = m_pDevice->GetInitInfo().pPalDevice;

    Pal::Result result = Pal::Result::Success;
    void* pObject = PAL_MALLOC(pPalDevice->GetGpuMemorySize(createInfo, &result),
                               m_pDevice,
                               Util::SystemAllocType::AllocObject);

    if ((result == Pal::Result::Success) && (pObject == nullptr))
    {
        result = Pal::Result::ErrorOutOfMemory;
    }

    Pal::IGpuMemory* pGpuMem = nullptr;
    if (result == Pal::Result::Success)
    {
        result = pPalDevice->CreateGpuMemory(createInfo, pObject, &pGpuMem);
    }

    if (result == Pal::Result::Success)
    {
        Pal::GpuMemoryRef ref = {};
        ref.pGpuMemory = pGpuMem;

        result = pPalDevice->AddGpuMemoryReferences(1, &ref, nullptr, Pal::GpuMemoryRefMustSucceed);

        if (result != Pal::Result::Success)
        {
            pGpuMem->Destroy();
            pGpuMem = nullptr;
        }
    }

    if ((pGpuMem == nullptr) && (pObject != nullptr))
    {
        PAL_FREE(pObject, m_pDevice);
    }

    *ppGpuMem = pGpuMem;

    return result;
}

// =====================================================================================================================
void BlasStatsLog::DestroyGpuMemory(
    Pal::IGpuMemory** ppGpuMem)
{
    Pal::IGpuMemory* pGpuMem = *ppGpuMem;

    if (pGpuMem != nullptr)
    {
        m_pDevice->GetInitInfo().pPalDevice->RemoveGpuMemoryReferences(1, &pGpuMem, nullptr);
        pGpuMem->Destroy();
        PAL_FREE(pGpuMem, m_pDevice);
        *ppGpuMem = nullptr;
    }
}

// =====================================================================================================================
void BlasStatsLog::Init()
{
    const DeviceSettings& settings = m_pDevice->Settings();

    m_pFile = fopen(settings.blasStatsLogPath, "a");

    if (m_pFile == nullptr)
    {
        fprintf(stderr, "GPURT BLAS stats log: cannot open %s\n", settings.blasStatsLogPath);
        return;
    }

    // Ring: same memory setup as the GPU debug monitor, bypassing GPU L2 so the CPU sees the GPU writes.
    Pal::GpuMemoryCreateInfo ringInfo = {};
    ringInfo.flags.gl2Uncached = 1;
    ringInfo.size              = RingSizeBytes;
    ringInfo.heapAccess        = Pal::GpuHeapAccessCpuReadMostly;

    // Accumulators: GPU-local, so the counting atomics are L2 atomics instead of PCIe atomics.
    Pal::GpuMemoryCreateInfo accumInfo = {};
    accumInfo.size       = RingEntries * BLAS_STATS_ENTRY_SIZE;
    accumInfo.heapAccess = Pal::GpuHeapAccessGpuMostly;

    void* pMapped = nullptr;

    Pal::Result result = CreateGpuMemory(ringInfo, &m_pGpuMem);

    if (result == Pal::Result::Success)
    {
        result = m_pGpuMem->Map(&pMapped);
    }

    if (result == Pal::Result::Success)
    {
        result = CreateGpuMemory(accumInfo, &m_pAccumMem);
    }

    if (result != Pal::Result::Success)
    {
        fprintf(stderr, "GPURT BLAS stats log: buffer allocation failed (%d)\n", int(result));

        DestroyGpuMemory(&m_pGpuMem);
        DestroyGpuMemory(&m_pAccumMem);
        fclose(m_pFile);
        m_pFile = nullptr;
        return;
    }

    memset(pMapped, 0, RingSizeBytes);

    m_pRing = static_cast<uint32*>(pMapped);
    m_pid   = Util::GetIdOfCurrentProcess();

    WriteFileHeader();

    if (m_thread.Begin(&MonitorThread, this) == Util::Result::Success)
    {
        m_threadStarted = true;
    }
}

// =====================================================================================================================
// One comment block per logical device, then the column header.
void BlasStatsLog::WriteFileHeader()
{
    const DeviceSettings& settings = m_pDevice->Settings();

    char  exePath[512] = {};
    char* pExeName     = nullptr;
    if (Util::GetExecutableName(exePath, &pExeName, sizeof(exePath)) != Util::Result::Success)
    {
        pExeName = nullptr;
    }

    char now[64] = {};
    FormatNow(now, sizeof(now));

    char arguments[1024] = {};
    FormatArguments(arguments, sizeof(arguments));

    uint32 wideNodeMergingFlags = 0;
    uint32 obbFlags             = 0;
    uint32 primCompMode         = 0;
#if GPURT_BUILD_RTIP3_1
    wideNodeMergingFlags = settings.wideNodeMergingFlags;
    obbFlags             = settings.enableOrientedBoundingBoxes;
    primCompMode         = settings.primCompressionFlags;
#endif

#if defined(SOGANG_EXPERIMENT) && SOGANG_EXPERIMENT
    constexpr uint32 SogangExperiment = 1;
#else
    constexpr uint32 SogangExperiment = 0;
#endif

    fprintf(m_pFile,
            "# gpurt BLAS stats log: process=%s exe=%s pid=%u start=%s\n",
            (pExeName != nullptr) ? pExeName : "?",
            exePath,
            m_pid,
            now);
    fprintf(m_pFile, "# args: %s\n", arguments);
    fprintf(m_pFile,
            "# settings: WideNodeMergingFlags=0x%x EnableOrientedBoundingBoxes=0x%x RtPrimCompressionFlags=%u "
            "SOGANG_EXPERIMENT=%u\n",
            wideNodeMergingFlags,
            obbFlags,
            primCompMode,
            SogangExperiment);
    fprintf(m_pFile,
            "time,pid,seq,blasVA,numPrims,mergeFlags,obbFlags,pre,post,leafOnly,dobb,dobbLeafOnly,"
            "primStructs,compactedBytes,primComp,numGeoms,buildFlags,badBoxSlots\n");
    fflush(m_pFile);
}

// =====================================================================================================================
BlasStatsSlot BlasStatsLog::AllocateSlot()
{
    PAL_ASSERT(IsActive());

    uint32 seq = m_lastSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    if (seq == 0)
    {
        // 0 means "not done" in the marker array.
        seq = m_lastSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    const uint32  slotIndex = (seq - 1) % RingEntries;
    const gpusize baseVa    = m_pGpuMem->Desc().gpuVirtAddr;

    BlasStatsSlot slot = {};
    slot.seq      = seq;
    slot.markerVa = baseVa + (slotIndex * sizeof(uint32));
    slot.entryVa  = baseVa + MarkerBytes + (slotIndex * BLAS_STATS_ENTRY_SIZE);
    slot.accumVa  = m_pAccumMem->Desc().gpuVirtAddr + (slotIndex * BLAS_STATS_ENTRY_SIZE);

    return slot;
}

// =====================================================================================================================
// Writes every completed slot to the log file, in sequence order, and frees it for the next build.
void BlasStatsLog::Drain()
{
    struct Record
    {
        uint32 dw[BLAS_STATS_ENTRY_DWORDS];
    };

    const uint32 lastSeq   = m_lastSeq.load(std::memory_order_relaxed);
    const uint32 scanCount = Util::Min(lastSeq, RingEntries);

    volatile uint32* pMarkers = m_pRing;
    volatile uint32* pEntries = m_pRing + (MarkerBytes / sizeof(uint32));

    std::vector<Record> records;

    for (uint32 slotIndex = 0; slotIndex < scanCount; slotIndex++)
    {
        if (pMarkers[slotIndex] == 0)
        {
            continue;
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        Record record = {};
        const volatile uint32* pEntry = pEntries + (slotIndex * BLAS_STATS_ENTRY_DWORDS);
        for (uint32 i = 0; i < BLAS_STATS_ENTRY_DWORDS; i++)
        {
            record.dw[i] = pEntry[i];
        }

        std::atomic_thread_fence(std::memory_order_release);
        pMarkers[slotIndex] = 0;

        records.push_back(record);
    }

    if (records.empty())
    {
        return;
    }

    std::sort(records.begin(), records.end(), [](const Record& a, const Record& b)
    {
        return a.dw[BLAS_STATS_ENTRY_SEQ] < b.dw[BLAS_STATS_ENTRY_SEQ];
    });

    char now[64] = {};
    FormatNow(now, sizeof(now));

    uint32 primCompMode = 0;
#if GPURT_BUILD_RTIP3_1
    primCompMode = m_pDevice->Settings().primCompressionFlags;
#endif

    for (const Record& record : records)
    {
        const uint32* pDw       = record.dw;
        const uint64  blasVa    = (static_cast<uint64>(pDw[BLAS_STATS_ENTRY_BLAS_VA_HI]) << 32) |
                                  pDw[BLAS_STATS_ENTRY_BLAS_VA_LO];
        const uint32  post      = pDw[BLAS_STATS_ENTRY_POST];
        const uint32  followers = pDw[BLAS_STATS_ENTRY_FOLLOWERS];
        const uint32  buildInfo = pDw[BLAS_STATS_ENTRY_BUILD_INFO];

        const uint32 buildFlags   = buildInfo & 0xffff;
        const uint32 geometryType = buildInfo >> BLAS_STATS_BUILD_INFO_GEOMETRY_TYPE_SHIFT;
        const uint32 badBoxSlots  = pDw[BLAS_STATS_ENTRY_BAD_BOX_SLOTS];

        uint32 primComp = 0;
#if GPURT_BUILD_RTIP3_1
        primComp = GetEffectivePrimCompFlags(primCompMode, buildFlags, geometryType);
#endif

        fprintf(m_pFile,
                "%s,%u,%u,0x%llx,%u,0x%x,0x%x,%u,%u,%u,%u,%u,%u,%u,0x%x,%u,0x%x,%u\n",
                now,
                m_pid,
                pDw[BLAS_STATS_ENTRY_SEQ],
                static_cast<unsigned long long>(blasVa),
                pDw[BLAS_STATS_ENTRY_NUM_PRIMS],
                pDw[BLAS_STATS_ENTRY_MERGE_FLAGS],
                pDw[BLAS_STATS_ENTRY_OBB_FLAGS],
                post + followers,
                post,
                pDw[BLAS_STATS_ENTRY_LEAF_ONLY],
                pDw[BLAS_STATS_ENTRY_DOBB],
                pDw[BLAS_STATS_ENTRY_DOBB_LEAF_ONLY],
                pDw[BLAS_STATS_ENTRY_PRIM_STRUCTS],
                pDw[BLAS_STATS_ENTRY_COMPACTED_BYTES],
                primComp,
                pDw[BLAS_STATS_ENTRY_NUM_GEOMETRIES],
                buildFlags,
                badBoxSlots);

        if (badBoxSlots != 0)
        {
            // Box slots with the invalid bounds pattern: their subtrees are unreachable, so this BLAS is corrupt.
            fprintf(m_pFile,
                    "# warning: seq %u has %u used box slots with invalid bounds\n",
                    pDw[BLAS_STATS_ENTRY_SEQ],
                    badBoxSlots);
        }
    }

    fflush(m_pFile);
}

// =====================================================================================================================
void BlasStatsLog::MonitorThread(
    void* pParam)
{
    BlasStatsLog* pLog = static_cast<BlasStatsLog*>(pParam);

    while (true)
    {
        const bool exiting = pLog->m_exiting.load();

        pLog->Drain();

        if (exiting)
        {
            break;
        }

        Util::Sleep(std::chrono::milliseconds(10));
    }
}

} // namespace GpuRt
