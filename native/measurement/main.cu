#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static int64_t now() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC, &t)) throw std::runtime_error("clock_gettime failed");
    return int64_t(t.tv_sec) * 1000000000LL + t.tv_nsec;
}
static uint32_t number(const char* s, uint32_t lo, uint32_t hi) {
    std::string text(s);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected unsigned decimal argument");
    auto value = std::stoull(text);
    if (value < lo || value > hi) throw std::runtime_error("Argument outside calibration safety bounds");
    return static_cast<uint32_t>(value);
}
__global__ void transform(const uint32_t* input, uint32_t* output, uint32_t n, uint32_t rounds) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = input[i];
    for (uint32_t j = 0; j < rounds; ++j) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
    }
    output[i] = x;
}
struct Device {
    uint32_t *in = nullptr, *out = nullptr;
    cudaEvent_t first = nullptr, last = nullptr;
    ~Device() { if(first) cudaEventDestroy(first); if(last) cudaEventDestroy(last);
                if(in) cudaFree(in); if(out) cudaFree(out); }
};
static void await_command(const std::string& expected) {
    std::string line;
    if (!std::getline(std::cin, line) || line != expected) throw std::runtime_error("Invalid host protocol command");
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: atlas_measure ELEMENTS REPETITIONS ROUNDS SEED");
        const auto n = number(argv[1], 1, 1048576), repetitions = number(argv[2], 1, 1000000);
        const auto rounds = number(argv[3], 1, 4096), seed = number(argv[4], 1, 4294967295U);
        const auto prep_start = now();
        int devices = 0; check(cudaGetDeviceCount(&devices));
        if (devices != 1) throw std::runtime_error("Exactly one CUDA GPU required");
        cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
        char pci[32]; check(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0));
        std::vector<uint32_t> input(n), output(n), reference(n);
        for (uint32_t i = 0; i < n; ++i) {
            input[i] = i ^ seed;
            // Independent reference uses explicitly masked 64-bit arithmetic.
            uint64_t x = input[i];
            for (uint32_t j = 0; j < rounds; ++j) {
                x = (x ^ (x << 13)) & 0xffffffffULL;
                x = (x ^ (x >> 17)) & 0xffffffffULL;
                x = (x ^ (x << 5)) & 0xffffffffULL;
            }
            reference[i] = static_cast<uint32_t>(x);
        }
        Device d;
        const size_t bytes = size_t(n) * sizeof(uint32_t);
        check(cudaMalloc(&d.in, bytes)); check(cudaMalloc(&d.out, bytes));
        check(cudaEventCreate(&d.first)); check(cudaEventCreate(&d.last));
        check(cudaMemcpy(d.in, input.data(), bytes, cudaMemcpyHostToDevice));
        const auto warm_start = now();
        for (int k = 0; k < 8; ++k) transform<<<(n+255)/256, 256>>>(d.in, d.out, n, rounds);
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        const auto warm_end = now();
        std::cout << "{\"kind\":\"ready\",\"clock\":\"CLOCK_MONOTONIC\",\"preparation_start_ns\":" << prep_start
                  << ",\"warmup_start_ns\":" << warm_start << ",\"warmup_end_ns\":" << warm_end
                  << ",\"pci_bus_id\":\"" << pci << "\",\"compute_capability\":\"" << prop.major << "." << prop.minor
                  << "\"}" << std::endl;
        await_command("run");
        const auto start = now();
        check(cudaMemcpy(d.in, input.data(), bytes, cudaMemcpyHostToDevice));
        const auto h2d_end = now();
        check(cudaEventRecord(d.first));
        for (uint32_t k = 0; k < repetitions; ++k) transform<<<(n+255)/256, 256>>>(d.in, d.out, n, rounds);
        check(cudaGetLastError()); check(cudaEventRecord(d.last));
        check(cudaEventSynchronize(d.last));
        const auto compute_end = now();
        check(cudaMemcpy(output.data(), d.out, bytes, cudaMemcpyDeviceToHost));
        check(cudaDeviceSynchronize());
        const auto end = now();
        float ms = 0; check(cudaEventElapsedTime(&ms, d.first, d.last));
        std::cout << "{\"kind\":\"complete\",\"start_ns\":" << start << ",\"h2d_end_ns\":" << h2d_end
                  << ",\"compute_end_ns\":" << compute_end << ",\"end_ns\":" << end
                  << ",\"gpu_event_ms\":" << ms << ",\"elements\":" << n << ",\"repetitions\":" << repetitions
                  << ",\"rounds\":" << rounds << ",\"seed\":" << seed << "}" << std::endl;
        // Host obtains the final energy endpoint before validation/cleanup starts.
        await_command("validate");
        size_t errors = 0;
        for (uint32_t i = 0; i < n; ++i) if (output[i] != reference[i]) ++errors;
        std::cout << "{\"kind\":\"validation\",\"status\":\"" << (errors ? "failed" : "passed")
                  << "\",\"mismatches\":" << errors << ",\"elements_verified\":" << n
                  << ",\"oracle\":\"masked_uint64_xorshift\",\"tolerance\":0}" << std::endl;
        return errors ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
