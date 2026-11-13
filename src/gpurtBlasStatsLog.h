/*
 ***********************************************************************************************************************
 *
 *  Per-BLAS node statistics log (DeviceSettings::enableBlasStatsLog).
 *
 *  When enabled on an RTIP 3.1 device, every BLAS build records a BlasStats3_1 dispatch that counts the box nodes of
 *  the finished BLAS into a slot of a persistently mapped ring buffer (layout: src/shared/gpurtBlasStats.h; the
 *  counting itself goes through a GPU-local accumulator). A background thread polls the ring and appends one CSV line
 *  per completed slot to blasStatsLogPath. When disabled nothing is allocated, no thread runs and no dispatch is
 *  recorded.
 *
 **********************************************************************************************************************/
#pragma once

#include "gpurt/gpurt.h"

#include "palGpuMemory.h"
#include "palThread.h"

#include <atomic>
#include <cstdio>

namespace GpuRt
{

namespace Internal
{
    class Device;
}

// A ring slot handed to one BLAS build
struct BlasStatsSlot
{
    uint32  seq;      // Sequence number, never 0
    gpusize entryVa;  // GPU VA of the 64-byte entry
    gpusize markerVa; // GPU VA of the done marker dword
    gpusize accumVa;  // GPU VA of the 64-byte GPU-local accumulator
};

class BlasStatsLog
{
public:
    explicit BlasStatsLog(Internal::Device* pDevice);
    ~BlasStatsLog();

    // Allocates the ring, opens the log file and starts the monitor thread. On failure the log stays inactive.
    void Init();

    bool IsActive() const { return m_pRing != nullptr; }

    // Thread-safe. Only valid when IsActive().
    BlasStatsSlot AllocateSlot();

private:
    void Drain();
    void WriteFileHeader();
    Pal::Result CreateGpuMemory(const Pal::GpuMemoryCreateInfo& createInfo, Pal::IGpuMemory** ppGpuMem);
    void DestroyGpuMemory(Pal::IGpuMemory** ppGpuMem);

    static void MonitorThread(void* pParam);

    Internal::Device*   m_pDevice;
    Pal::IGpuMemory*    m_pGpuMem;        // CPU-visible ring: markers, then entries
    Pal::IGpuMemory*    m_pAccumMem;      // GPU-local accumulators, one per slot
    uint32*             m_pRing;          // Mapped ring
    FILE*               m_pFile;
    uint32              m_pid;
    Util::Thread        m_thread;
    bool                m_threadStarted;
    std::atomic<bool>   m_exiting;
    std::atomic<uint32> m_lastSeq;        // Last allocated sequence number
};

} // namespace GpuRt
