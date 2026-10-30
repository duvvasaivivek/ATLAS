#include "backend.hpp"
#include <omp.h>
#include <nvtx3/nvToolsExt.h>
#include <stdexcept>
#include <time.h>

namespace atlas {
int64_t monotonic_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC, &t)) throw std::runtime_error("CLOCK_MONOTONIC failed");
    return int64_t(t.tv_sec)*1000000000LL+t.tv_nsec;
}
uint32_t reference_value(uint32_t input, uint32_t rounds) {
    uint64_t value = input;
    for (uint32_t j=0; j<rounds; ++j) {
        value = (value ^ (value << 13)) & 0xffffffffULL;
        value = (value ^ (value >> 17)) & 0xffffffffULL;
        value = (value ^ (value << 5)) & 0xffffffffULL;
    }
    return static_cast<uint32_t>(value);
}
CpuResult cpu_execute(const uint32_t* input, uint32_t* output, uint32_t* visits,
                      uint32_t count, uint32_t rounds, int workers) {
    CpuResult result;
    result.spans.resize(workers);
    result.start = monotonic_ns();
    uint64_t sum = 0;
    // No empty-team launch or division by zero at GPU-only endpoints.
    if (count) {
        #pragma omp parallel num_threads(workers) reduction(+:sum)
        {
            const int id = omp_get_thread_num(), team = omp_get_num_threads();
            #pragma omp single
            result.workers = team;
            const uint32_t lo = uint64_t(count)*id/team, hi = uint64_t(count)*(id+1)/team;
            if (lo < hi) {
                nvtxRangePushA("atlas_cpu_compute");
                result.spans[id].start = monotonic_ns();
                for (uint32_t i=lo; i<hi; ++i) {
                    uint32_t value = input[i];
                    for (uint32_t j=0; j<rounds; ++j) {
                        value ^= value << 13;
                        value ^= value >> 17;
                        value ^= value << 5;
                    }
                    output[i] = value;
                    ++visits[i];
                    sum += value;
                }
                result.spans[id].end = monotonic_ns();
                result.spans[id].items = hi-lo;
                nvtxRangePop();
            }
        }
    }
    result.sum = sum;
    result.end = monotonic_ns();
    return result;
}
}
