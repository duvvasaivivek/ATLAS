#include "backend.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <omp.h>
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static uint32_t argument(const char* raw, uint32_t low, uint32_t high) {
    std::string text(raw);
    if(text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected unsigned decimal integer");
    auto value = std::stoull(text);
    if(value < low || value > high) throw std::runtime_error("Argument outside supported limits");
    return static_cast<uint32_t>(value);
}

__global__ void atlas_static_transform_reduce(const uint32_t* input, uint32_t* output,
                                              uint32_t* visits, unsigned long long* partial,
                                              uint32_t count, uint32_t rounds) {
    __shared__ unsigned long long sums[256];
    const uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t value = 0;
    if(i < count) {
        value = input[i];
        for(uint32_t j=0; j<rounds; ++j) {
            value ^= value << 13;
            value ^= value >> 17;
            value ^= value << 5;
        }
        output[i] = value;
        atomicAdd(visits+i, 1U);
    }
    sums[threadIdx.x] = value;
    __syncthreads();
    for(unsigned step=128; step; step/=2) {
        if(threadIdx.x < step) sums[threadIdx.x] += sums[threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x == 0) partial[blockIdx.x] = sums[0];
}

// Reused for warm-up and every measured iteration. All sizes are bounded before allocation.
struct Buffers {
    uint32_t *input=nullptr, *gpu_output=nullptr, *gpu_visits=nullptr;
    uint32_t *device_input=nullptr, *device_output=nullptr, *device_visits=nullptr;
    unsigned long long *partial=nullptr, *device_partial=nullptr;
    cudaStream_t stream=nullptr;
    ~Buffers() {
        if(stream) cudaStreamDestroy(stream);
        if(device_input) cudaFree(device_input);
        if(device_output) cudaFree(device_output);
        if(device_visits) cudaFree(device_visits);
        if(device_partial) cudaFree(device_partial);
        if(input) cudaFreeHost(input);
        if(gpu_output) cudaFreeHost(gpu_output);
        if(gpu_visits) cudaFreeHost(gpu_visits);
        if(partial) cudaFreeHost(partial);
    }
    void allocate(uint32_t n) {
        const size_t bytes=size_t(n)*4, reduction_bytes=size_t((n+255)/256)*8;
        check(cudaMallocHost(&input,bytes)); check(cudaMallocHost(&gpu_output,bytes));
        check(cudaMallocHost(&gpu_visits,bytes)); check(cudaMallocHost(&partial,reduction_bytes));
        check(cudaMalloc(&device_input,bytes)); check(cudaMalloc(&device_output,bytes));
        check(cudaMalloc(&device_visits,bytes)); check(cudaMalloc(&device_partial,reduction_bytes));
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
};

// The native CPU worker outlives every iteration. Python merely waits for this process.
class CpuWorker {
    std::mutex mutex;
    std::condition_variable cv;
    bool pending=false, done=false, stopping=false;
    const uint32_t* input;
    uint32_t *output, *visits;
    uint32_t count, rounds;
    int workers;
    atlas::CpuResult result;
    std::exception_ptr failure;
    std::thread thread;
    void loop() {
        std::unique_lock<std::mutex> guard(mutex);
        for(;;) {
            cv.wait(guard,[&]{return pending || stopping;});
            if(stopping) return;
            pending=false;
            guard.unlock();
            try { result=atlas::cpu_execute(input,output,visits,count,rounds,workers); }
            catch(...) { failure=std::current_exception(); }
            guard.lock(); done=true; cv.notify_all();
        }
    }
public:
    CpuWorker(const uint32_t* in,uint32_t* out,uint32_t* seen,uint32_t n,uint32_t r,int w)
      : input(in),output(out),visits(seen),count(n),rounds(r),workers(w),thread(&CpuWorker::loop,this) {}
    ~CpuWorker() {
        { std::lock_guard<std::mutex> guard(mutex); stopping=true; cv.notify_all(); }
        thread.join();
    }
    void release() {
        std::lock_guard<std::mutex> guard(mutex);
        done=false; pending=true; failure=nullptr; cv.notify_all();
    }
    atlas::CpuResult wait() {
        std::unique_lock<std::mutex> guard(mutex);
        cv.wait(guard,[&]{return done;});
        if(failure) std::rethrow_exception(failure);
        return result;
    }
};

int main(int argc,char** argv) {
    try {
        if(argc != 7) throw std::runtime_error("Usage: atlas_static N CPU_ITEMS ROUNDS ITERATIONS WORKERS SEED");
        const auto n=argument(argv[1],1,1048579), cpu_n=argument(argv[2],0,n);
        const auto rounds=argument(argv[3],1,4096), iterations=argument(argv[4],1,100);
        const auto workers=argument(argv[5],1,8), seed=argument(argv[6],1,0xffffffffU);
        const auto gpu_n=n-cpu_n, blocks=(gpu_n+255)/256;
        const auto preparation_start=atlas::monotonic_ns();
        int devices=0; check(cudaGetDeviceCount(&devices));
        if(devices != 1) throw std::runtime_error("Exactly one CUDA GPU required");
        omp_set_dynamic(0);
        Buffers buffers; buffers.allocate(n);
        std::vector<uint32_t> output(n),visits(n),reference(n);
        uint64_t reference_sum=0;
        for(uint32_t i=0;i<n;++i) {
            buffers.input[i]=i^seed;
            reference[i]=atlas::reference_value(buffers.input[i],rounds);
            reference_sum+=reference[i];
        }
        CpuWorker worker(buffers.input,output.data(),visits.data(),cpu_n,rounds,workers);
        std::cout << "{\"kind\":\"configuration\",\"schema_version\":\"3.0\",\"clock\":\"CLOCK_MONOTONIC\","
                  << "\"elements\":" << n << ",\"cpu_items\":" << cpu_n << ",\"gpu_items\":" << gpu_n
                  << ",\"rounds\":" << rounds << ",\"iterations\":" << iterations << ",\"workers_requested\":" << workers
                  << ",\"seed\":" << seed << ",\"preparation_start_ns\":" << preparation_start
                  << ",\"reference_sum\":" << reference_sum << ",\"oracle\":\"masked_uint64_xorshift\",\"tolerance\":0,"
                  << "\"buffer_capacity_items\":" << n << ",\"allocation_policy\":\"allocate_once_reuse\"}" << std::endl;
        for(uint32_t iteration=0;iteration<=iterations;++iteration) {
            const auto reset_start=atlas::monotonic_ns();
            std::fill(output.begin(),output.end(),0xa5a5a5a5U);
            std::fill(visits.begin(),visits.end(),0U);
            if(gpu_n) {
                check(cudaMemsetAsync(buffers.device_output,0xa5,size_t(gpu_n)*4,buffers.stream));
                check(cudaMemsetAsync(buffers.device_visits,0,size_t(gpu_n)*4,buffers.stream));
                check(cudaStreamSynchronize(buffers.stream));
            }
            const auto reset_end=atlas::monotonic_ns();
            nvtxRangePushA(iteration ? "atlas_iteration" : "atlas_warmup");
            const auto release=atlas::monotonic_ns();
            if(cpu_n) worker.release();
            const auto gpu_dispatch=atlas::monotonic_ns();
            int64_t h2d_end=gpu_dispatch,compute_end=gpu_dispatch,gpu_end=gpu_dispatch;
            if(gpu_n) {
                nvtxRangePushA("atlas_gpu_branch");
                check(cudaMemcpyAsync(buffers.device_input,buffers.input+cpu_n,size_t(gpu_n)*4,cudaMemcpyHostToDevice,buffers.stream));
                check(cudaStreamSynchronize(buffers.stream)); h2d_end=atlas::monotonic_ns();
                atlas_static_transform_reduce<<<blocks,256,0,buffers.stream>>>(buffers.device_input,buffers.device_output,buffers.device_visits,buffers.device_partial,gpu_n,rounds);
                check(cudaGetLastError()); check(cudaStreamSynchronize(buffers.stream)); compute_end=atlas::monotonic_ns();
                check(cudaMemcpyAsync(buffers.gpu_output,buffers.device_output,size_t(gpu_n)*4,cudaMemcpyDeviceToHost,buffers.stream));
                check(cudaMemcpyAsync(buffers.gpu_visits,buffers.device_visits,size_t(gpu_n)*4,cudaMemcpyDeviceToHost,buffers.stream));
                check(cudaMemcpyAsync(buffers.partial,buffers.device_partial,size_t(blocks)*8,cudaMemcpyDeviceToHost,buffers.stream));
                check(cudaStreamSynchronize(buffers.stream)); gpu_end=atlas::monotonic_ns();
                nvtxRangePop();
            }
            atlas::CpuResult cpu;
            if(cpu_n) cpu=worker.wait();
            const auto barrier_end=atlas::monotonic_ns();
            nvtxRangePushA("atlas_merge");
            uint64_t gpu_sum=0;
            if(gpu_n) {
                std::copy(buffers.gpu_output,buffers.gpu_output+gpu_n,output.data()+cpu_n);
                std::copy(buffers.gpu_visits,buffers.gpu_visits+gpu_n,visits.data()+cpu_n);
                for(uint32_t i=0;i<blocks;++i) gpu_sum+=buffers.partial[i];
            }
            const uint64_t combined_sum=cpu.sum+gpu_sum;
            const auto merge_end=atlas::monotonic_ns();
            nvtxRangePop(); nvtxRangePop();
            size_t mismatches=0,coverage_errors=0;
            for(uint32_t i=0;i<n;++i) {
                mismatches+=(output[i]!=reference[i]); coverage_errors+=(visits[i]!=1);
            }
            const auto validation_end=atlas::monotonic_ns();
            const bool passed=!mismatches && !coverage_errors && combined_sum==reference_sum;
            std::cout << "{\"kind\":\"iteration\",\"iteration\":" << iteration << ",\"warmup\":" << (iteration ? "false":"true")
                      << ",\"status\":\"" << (passed ? "passed":"failed") << "\",\"reset_start_ns\":" << reset_start
                      << ",\"reset_end_ns\":" << reset_end << ",\"release_ns\":" << release
                      << ",\"cpu_start_ns\":" << (cpu_n ? std::to_string(cpu.start):"null")
                      << ",\"cpu_end_ns\":" << (cpu_n ? std::to_string(cpu.end):"null")
                      << ",\"gpu_dispatch_ns\":" << (gpu_n ? std::to_string(gpu_dispatch):"null")
                      << ",\"h2d_end_ns\":" << (gpu_n ? std::to_string(h2d_end):"null")
                      << ",\"compute_end_ns\":" << (gpu_n ? std::to_string(compute_end):"null")
                      << ",\"gpu_end_ns\":" << (gpu_n ? std::to_string(gpu_end):"null")
                      << ",\"barrier_end_ns\":" << barrier_end << ",\"merge_end_ns\":" << merge_end
                      << ",\"validation_end_ns\":" << validation_end << ",\"cpu_sum\":" << cpu.sum
                      << ",\"gpu_sum\":" << gpu_sum << ",\"combined_sum\":" << combined_sum
                      << ",\"mismatches\":" << mismatches << ",\"coverage_errors\":" << coverage_errors
                      << ",\"elements_verified\":" << n << ",\"workers_used\":" << cpu.workers << ",\"worker_spans\":[";
            for(size_t i=0;i<cpu.spans.size();++i) {
                if(i) std::cout << ',';
                const auto& s=cpu.spans[i];
                std::cout << "{\"worker\":" << i << ",\"start_ns\":" << s.start << ",\"end_ns\":" << s.end << ",\"items\":" << s.items << '}';
            }
            std::cout << "]}" << std::endl;
            if(!passed) return 1;
        }
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
