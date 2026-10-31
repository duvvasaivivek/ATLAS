#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static int64_t now() { timespec t{}; if(clock_gettime(CLOCK_MONOTONIC,&t)) throw std::runtime_error("clock failed"); return int64_t(t.tv_sec)*1000000000LL+t.tv_nsec; }
static uint32_t number(const std::string& s,uint32_t lo,uint32_t hi) {
    if(s.empty() || s.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("Unsigned decimal required");
    auto v=std::stoull(s); if(v<lo || v>hi) throw std::runtime_error("Argument outside probe bounds"); return uint32_t(v);
}
__global__ void compute(const uint32_t* in,uint32_t* out,uint32_t n,uint32_t rounds) {
    auto i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t x=in[i];
    for(uint32_t r=0;r<rounds;++r) { x^=x<<13; x^=x>>17; x^=x<<5; }
    out[i]=x;
}
__global__ void stream(const uint32_t* in,uint32_t* out,uint32_t n) {
    auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=in[i]*1664525U+1013904223U;
}
struct Device { uint32_t *in=nullptr,*out=nullptr; ~Device(){ if(in)cudaFree(in);if(out)cudaFree(out); } };
int main(int argc,char** argv) {
    try {
        if(argc!=5)throw std::runtime_error("Usage: atlas_frequency_probe compute|stream ELEMENTS ROUNDS SEED; stdin: run REPETITIONS / quit");
        std::string mode=argv[1]; if(mode!="compute" && mode!="stream")throw std::runtime_error("Invalid mode");
        auto n=number(argv[2],1,16777216), rounds=number(argv[3],1,4096), seed=number(argv[4],1,4294967295U);
        if(mode=="compute" && uint64_t(n)*rounds>268435456ULL)throw std::runtime_error("Compute reference budget exceeded");
        if(mode=="stream" && rounds!=1)throw std::runtime_error("Stream mode requires ROUNDS=1");
        int count=0;check(cudaGetDeviceCount(&count));if(count!=1)throw std::runtime_error("Exactly one CUDA GPU required");
        char pci[32];check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));
        std::vector<uint32_t> input(n),output(n),reference(n);
        for(uint32_t i=0;i<n;++i) {
            input[i]=i^seed;
            uint64_t x=input[i];
            if(mode=="compute") for(uint32_t r=0;r<rounds;++r) {
                x=(x^(x<<13))&0xffffffffULL; x=(x^(x>>17))&0xffffffffULL; x=(x^(x<<5))&0xffffffffULL;
            } else x=(x*1664525ULL+1013904223ULL)&0xffffffffULL;
            reference[i]=uint32_t(x);
        }
        Device d;size_t bytes=size_t(n)*sizeof(uint32_t);
        check(cudaMalloc(&d.in,bytes));check(cudaMalloc(&d.out,bytes));
        check(cudaMemcpy(d.in,input.data(),bytes,cudaMemcpyHostToDevice));
        auto launch=[&](){ if(mode=="compute")compute<<<(n+255)/256,256>>>(d.in,d.out,n,rounds);
                          else stream<<<(n+255)/256,256>>>(d.in,d.out,n);check(cudaGetLastError()); };
        auto warm_start=now();launch();check(cudaDeviceSynchronize());auto warm_end=now();
        std::cout<<"{\"kind\":\"ready\",\"clock\":\"CLOCK_MONOTONIC\",\"mode\":\""<<mode
                 <<"\",\"elements\":"<<n<<",\"rounds\":"<<rounds<<",\"seed\":"<<seed
                 <<",\"pci_bus_id\":\""<<pci<<"\",\"warmup_start_ns\":"<<warm_start<<",\"warmup_end_ns\":"<<warm_end<<"}"<<std::endl;
        std::string line;uint32_t batch=0;
        while(std::getline(std::cin,line)) {
            if(line=="quit")return 0;
            std::istringstream command(line);std::string verb,reps,extra;
            if(!(command>>verb>>reps) || verb!="run" || (command>>extra))throw std::runtime_error("Invalid command");
            auto repetitions=number(reps,1,10000);
            auto start=now();for(uint32_t r=0;r<repetitions;++r)launch();check(cudaDeviceSynchronize());auto compute_end=now();
            check(cudaMemcpy(output.data(),d.out,bytes,cudaMemcpyDeviceToHost));auto download_end=now();
            size_t mismatches=0;uint64_t checksum=0;
            for(uint32_t i=0;i<n;++i){mismatches+=output[i]!=reference[i];checksum+=output[i];}
            auto validation_end=now();
            std::cout<<"{\"kind\":\"batch\",\"batch\":"<<batch++<<",\"mode\":\""<<mode
                     <<"\",\"elements\":"<<n<<",\"rounds\":"<<rounds<<",\"seed\":"<<seed
                     <<",\"repetitions\":"<<repetitions<<",\"start_ns\":"<<start<<",\"compute_end_ns\":"<<compute_end
                     <<",\"download_end_ns\":"<<download_end<<",\"validation_end_ns\":"<<validation_end
                     <<",\"mismatches\":"<<mismatches<<",\"elements_verified\":"<<n<<",\"checksum\":"<<checksum
                     <<",\"oracle\":\"masked_uint64_reference\",\"tolerance\":0}"<<std::endl;
            if(mismatches)return 2;
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
