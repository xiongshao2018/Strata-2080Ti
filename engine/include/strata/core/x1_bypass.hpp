#pragma once
// include/strata/core/x1_bypass.hpp - the two-stage host->GPU0->GPU1 copy for expert blobs (STRATA_X1_BYPASS).
//
// Two RTX 2080 Ti joined by an NVLink bridge, but the second card sits behind
// a chipset root port that negotiates x1 (~0.8 GB/s) while the first is x16
// (~12.4 GB/s).  A streamed expert blob for the second card is normally DMA'd
// straight from pinned host RAM across that x1 link.  With STRATA_X1_BYPASS=1
// the same bytes go host -> GPU0 over x16, then GPU0 -> GPU1 over NVLink
// (cudaMemcpyPeerAsync), and the caller's copy stream waits on the peer copy.
// Unset or 0 keeps the original single H2D byte-for-byte.
//
// One staging buffer and one stream on GPU0.  All callers share one FIFO
// stream, so one event is enough for the caller's wait.  The peer copy is the
// operation that actually writes the destination, so a caller whose slot is
// still being read must pass that slot's release event: a wait on the caller's
// copy stream cannot stop x.s from overwriting memory the compute stream is
// still reading.
//
// Shared by the prompt path's streamed experts (prefill.cpp) and the expert
// cache's queued refills (expert_cache.cpp): both copy the same blobs into the
// same kind of VRAM slot, and a refill that took the direct H2D path paid the
// x1 link for every lent slot.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "strata/core/on_device.hpp"

namespace strata::core {

#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL) || defined(SYCL_LANGUAGE_VERSION)

inline bool x1_bypass_on() { return false; }

// Drop-in replacement for
//   cudaMemcpyAsync(dst, host, n, cudaMemcpyHostToDevice, s)
inline cudaError_t h2d_expert(uint8_t* dst, const uint8_t* host, size_t n, cudaStream_t s,
                              cudaEvent_t slot_release = nullptr) {
    (void) slot_release;
    return cudaMemcpyAsync(dst, host, n, cudaMemcpyHostToDevice, s);
}

#else

inline bool x1_bypass_on() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_X1_BYPASS");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}

struct X1Bypass {
    static constexpr int kMaxStages = 8;
    // One staging buffer, stream and event per stage.  With one stage the
    // host->GPU0 copy and the GPU0->GPU1 peer copy run back to back; with
    // several, stage i's peer copy overlaps stage i+1's host copy.
    struct Stage {
        cudaStream_t s = nullptr;
        cudaEvent_t ev = nullptr;
        uint8_t* buf = nullptr;
        size_t bytes = 0;
    };
    std::mutex mu;
    Stage st[kMaxStages];
    int stages = 0;   // STRATA_X1_STAGES (1..8, default 2)
    int next = 0;
    bool failed = false;
    bool warned = false;
    bool logged = false;
    bool peers_tried = false;

    ~X1Bypass() {
        for (auto& x : st) {
            if (x.s) cudaStreamDestroy(x.s);
            if (x.ev) cudaEventDestroy(x.ev);
            if (x.buf) cudaFree(x.buf);
        }
    }
    static int stages_env() {
        const char* e = std::getenv("STRATA_X1_STAGES");
        const int n = e != nullptr && e[0] != '\0' ? std::atoi(e) : 2;
        return std::clamp(n, 1, kMaxStages);
    }
    bool ensure(size_t n) {
        if (failed) return false;
        const strata::core::OnDevice on(0);
        if (stages == 0) stages = stages_env();
        if (!peers_tried) {
            peers_tried = true;
            int a = 0, b = 0;
            cudaDeviceCanAccessPeer(&a, 0, 1);
            if (a) {
                cudaError_t e0 = cudaDeviceEnablePeerAccess(1, 0);
                if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
                const strata::core::OnDevice on1(1);
                cudaError_t e1 = cudaDeviceEnablePeerAccess(0, 0);
                if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
                b = e0 == cudaSuccess && e1 == cudaSuccess;
            }
            if (!a || !b)
                std::fprintf(stderr, "strata: x1 bypass: GPU0/GPU1 peer access is not available; the two-stage copy may stage through the host\n");
        }
        for (int i = 0; i < stages; ++i) {
            Stage& x = st[i];
            if (!x.s && cudaStreamCreateWithFlags(&x.s, cudaStreamNonBlocking) != cudaSuccess) { failed = true; return false; }
            if (!x.ev && cudaEventCreateWithFlags(&x.ev, cudaEventDisableTiming) != cudaSuccess) { failed = true; return false; }
            if (x.buf && x.bytes < n) {
                // x.s is non-blocking: cudaFree's implicit sync does not drain
                // the peer copies queued on it.  Wait before releasing x.buf.
                cudaStreamSynchronize(x.s);
                cudaFree(x.buf); x.buf = nullptr; x.bytes = 0;
            }
            if (!x.buf) {
                const size_t want = n + 256;
                if (cudaMalloc((void**) &x.buf, want) != cudaSuccess) { x.buf = nullptr; failed = true; return false; }
                x.bytes = want;
            }
        }
        return true;
    }
};

inline X1Bypass& x1_bypass() { static X1Bypass v; return v; }

// Drop-in replacement for
//   cudaMemcpyAsync(dst, host, n, cudaMemcpyHostToDevice, s)
// for a streamed expert blob.  Any failure falls back to that exact call.
inline cudaError_t h2d_expert(uint8_t* dst, const uint8_t* host, size_t n, cudaStream_t s,
                              cudaEvent_t slot_release = nullptr) {
    if (!x1_bypass_on()) {
        return cudaMemcpyAsync(dst, host, n, cudaMemcpyHostToDevice, s);
    }
    cudaPointerAttributes at{};
    if (cudaPointerGetAttributes(&at, dst) != cudaSuccess || at.device <= 0) {
        return cudaMemcpyAsync(dst, host, n, cudaMemcpyHostToDevice, s);
    }
    X1Bypass& x = x1_bypass();
    std::lock_guard<std::mutex> lock(x.mu);
    if (!x.ensure(n)) {
        if (!x.warned) {
            x.warned = true;
            std::fprintf(stderr, "strata: x1 bypass: GPU0 staging setup failed; falling back to the direct host-to-device copy\n");
        }
        return cudaMemcpyAsync(dst, host, n, cudaMemcpyHostToDevice, s);
    }
    if (x.next >= x.stages) x.next = 0;
    X1Bypass::Stage& g = x.st[x.next];
    x.next = (x.next + 1) % x.stages;
    {
        const strata::core::OnDevice on(0);
        cudaError_t e = cudaMemcpyAsync(g.buf, host, n, cudaMemcpyHostToDevice, g.s);
        if (e == cudaSuccess && slot_release) e = cudaStreamWaitEvent(g.s, slot_release, 0);
        if (e == cudaSuccess) e = cudaMemcpyPeerAsync(dst, at.device, g.buf, 0, n, g.s);
        if (e == cudaSuccess) e = cudaEventRecord(g.ev, g.s);
        if (e != cudaSuccess) {
            if (!x.warned) {
                x.warned = true;
                std::fprintf(stderr, "strata: x1 bypass: the two-stage copy failed; falling back to the direct host-to-device copy\n");
            }
            return cudaMemcpyAsync(dst, host, n, cudaMemcpyHostToDevice, s);
        }
        if (!x.logged) {
            x.logged = true;
            std::fprintf(stderr, "strata: x1 bypass: two-stage expert copy active (host -> GPU0 staging -> device %d, %d stage buffers)\n",
                         at.device, x.stages);
        }
    }
    return cudaStreamWaitEvent(s, g.ev, 0);
}

#endif

}  // namespace strata::core
