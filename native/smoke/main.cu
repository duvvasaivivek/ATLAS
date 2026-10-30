#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int cpu_compute(const float*, const float*, float*, std::size_t);

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(std::size_t bytes) {
        check(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void gpu_compute(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = a[i] + 3.0f * b[i];
}

int main() {
    try {
        int count = 0, driver = 0, runtime = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count != 1) throw std::runtime_error("Phase 1 fixture requires exactly one GPU");
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        const std::vector<std::size_t> sizes = {1, 255, 256, 257, 4099, 1048579};
        std::size_t tested = 0;
        int workers = 0;
        for (const auto n : sizes) {
            std::vector<float> a(n), b(n), cpu(n, -1.0f), gpu(n, -1.0f);
            for (std::size_t i = 0; i < n; ++i) {
                a[i] = static_cast<float>(i % 251);
                b[i] = static_cast<float>((i * 7) % 127);
            }
            workers = cpu_compute(a.data(), b.data(), cpu.data(), n);
            const auto bytes = n * sizeof(float);
            DeviceBuffer da(bytes), db(bytes), dc(bytes);
            check(cudaMemcpy(da.ptr, a.data(), bytes, cudaMemcpyHostToDevice), "copy a to GPU");
            check(cudaMemcpy(db.ptr, b.data(), bytes, cudaMemcpyHostToDevice), "copy b to GPU");
            check(cudaMemset(dc.ptr, 0xff, bytes), "initialize output sentinel");
            gpu_compute<<<static_cast<unsigned>((n + 255) / 256), 256>>>(da.ptr, db.ptr, dc.ptr, n);
            check(cudaGetLastError(), "kernel launch");
            check(cudaDeviceSynchronize(), "kernel completion");
            check(cudaMemcpy(gpu.data(), dc.ptr, bytes, cudaMemcpyDeviceToHost), "copy result to CPU");
            for (std::size_t i = 0; i < n; ++i) {
                // Integer oracle is independent of both floating-point implementations.
                const float expected = static_cast<float>((i % 251) + 3 * ((i * 7) % 127));
                if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i]) ||
                    cpu[i] != expected || gpu[i] != expected) {
                    throw std::runtime_error("Result mismatch at size " + std::to_string(n) +
                                             " index " + std::to_string(i));
                }
            }
            tested += n;
        }
        std::cout << "{\"status\":\"passed\",\"device_name\":" << std::quoted(prop.name)
                  << ",\"compute_capability\":\"" << prop.major << "." << prop.minor << "\""
                  << ",\"cuda_driver_api_version\":" << driver
                  << ",\"cuda_runtime_version\":" << runtime
                  << ",\"device_memory_bytes\":" << prop.totalGlobalMem
                  << ",\"openmp_workers\":" << workers
                  << ",\"sizes\":[1,255,256,257,4099,1048579]"
                  << ",\"elements_verified_per_backend\":" << tested
                  << ",\"max_absolute_error\":0,\"comparison\":\"exact_against_integer_oracle\""
                  << ",\"cpu_gpu_concurrency_tested\":false,\"energy_measured\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU/CUDA smoke test failed: " << error.what() << '\n';
        return 1;
    }
}
