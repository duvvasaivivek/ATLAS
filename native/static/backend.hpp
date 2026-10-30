#pragma once
#include <cstdint>
#include <vector>

namespace atlas {
int64_t monotonic_ns();
struct WorkerSpan { int64_t start = 0, end = 0; uint32_t items = 0; };
struct CpuResult {
    int64_t start = 0, end = 0;
    uint64_t sum = 0;
    int workers = 0;
    std::vector<WorkerSpan> spans;
};
CpuResult cpu_execute(const uint32_t* input, uint32_t* output, uint32_t* visits,
                      uint32_t count, uint32_t rounds, int workers);
uint32_t reference_value(uint32_t input, uint32_t rounds);
}
