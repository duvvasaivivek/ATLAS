#include <cstddef>
#include <omp.h>

// This is a correctness fixture, not a workload or a performance benchmark.
int cpu_compute(const float* a, const float* b, float* out, std::size_t n) {
    int workers = 0;
    #pragma omp parallel
    {
        #pragma omp single
        workers = omp_get_num_threads();
        #pragma omp for
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = a[i] + 3.0f * b[i];
        }
    }
    return workers;
}
