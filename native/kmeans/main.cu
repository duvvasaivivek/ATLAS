#include "backend.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static unsigned number(const std::string& s,unsigned lo,unsigned hi) {
    if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("Expected unsigned integer");
    auto v=std::stoull(s); if(v<lo||v>hi) throw std::runtime_error("Argument outside supported range");
    return static_cast<unsigned>(v);
}
template<class T> static void array_json(const std::vector<T>& a) {
    std::cout << '['; for(size_t i=0;i<a.size();++i) {if(i)std::cout<<',';std::cout<<a[i];} std::cout<<']';
}
static std::string label_hash(const std::vector<uint32_t>& labels) {
    uint64_t value=14695981039346656037ULL;
    for(auto label:labels) for(unsigned j=0;j<4;++j) {value^=(label>>(8*j))&255U;value*=1099511628211ULL;}
    std::ostringstream s; s<<std::hex<<std::setfill('0')<<std::setw(16)<<value; return s.str();
}
class Worker {
    std::mutex mutex; std::condition_variable cv;
    bool pending=false,done=false,stop=false;
    std::function<void()> task; std::exception_ptr failure;
    std::thread thread;
    void loop() {
        std::unique_lock<std::mutex> lock(mutex);
        for(;;) {
            cv.wait(lock,[&]{return pending||stop;}); if(stop)return;
            pending=false; lock.unlock();
            try {task();} catch(...) {failure=std::current_exception();}
            lock.lock(); done=true; cv.notify_all();
        }
    }
public:
    explicit Worker(std::function<void()> f):task(std::move(f)),thread(&Worker::loop,this) {}
    ~Worker() {{std::lock_guard<std::mutex> lock(mutex);stop=true;cv.notify_all();}thread.join();}
    void release() {std::lock_guard<std::mutex> lock(mutex);done=false;pending=true;failure=nullptr;cv.notify_all();}
    void wait() {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return done;});if(failure)std::rethrow_exception(failure);}
};

__global__ void kmeans_assign(const double* points,const double* centers,uint32_t* labels,
                              double* sums,uint32_t* counts,double* objective,unsigned begin,unsigned n,unsigned k,unsigned d) {
    const unsigned i=begin+blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    double best=INFINITY; unsigned winner=0;
    for(unsigned c=0;c<k;++c) {
        double distance=0;
        for(unsigned j=0;j<d;++j) {double delta=points[size_t(i)*d+j]-centers[c*d+j];distance+=delta*delta;}
        if(distance<best) {best=distance;winner=c;}
    }
    labels[i]=winner; atomicAdd(counts+winner,1U); atomicAdd(objective,best);
    for(unsigned j=0;j<d;++j) atomicAdd(sums+winner*d+j,points[size_t(i)*d+j]);
}
// Block-local accumulation followed by a deterministic cross-block reduction.
// No point contends on a global objective/count/sum address in this path.
__global__ void kmeans_assign_reduced(const double* points,const double* centers,uint32_t* labels,
                                      double* partial,unsigned begin,unsigned n,unsigned k,unsigned d){
    extern __shared__ double storage[];
    const unsigned q=k*d+k,tid=threadIdx.x,i=begin+blockIdx.x*blockDim.x+tid;
    for(unsigned j=tid;j<q;j+=blockDim.x)storage[j]=0;
    double best=0;unsigned winner=0;
    __syncthreads();
    if(i<n){
        best=INFINITY;
        for(unsigned c=0;c<k;++c){double distance=0;
            for(unsigned j=0;j<d;++j){double delta=points[size_t(i)*d+j]-centers[c*d+j];distance+=delta*delta;}
            if(distance<best){best=distance;winner=c;}}
        labels[i]=winner;atomicAdd(storage+k*d+winner,1.);
        for(unsigned j=0;j<d;++j)atomicAdd(storage+winner*d+j,points[size_t(i)*d+j]);
    }
    storage[q+tid]=best;__syncthreads();
    for(unsigned step=128;step;step/=2){if(tid<step)storage[q+tid]+=storage[q+tid+step];__syncthreads();}
    for(unsigned j=tid;j<q;j+=blockDim.x)partial[size_t(blockIdx.x)*(q+1)+j]=storage[j];
    if(!tid)partial[size_t(blockIdx.x)*(q+1)+q]=storage[q];
}
__global__ void kmeans_finalize(const double* partial,double* sums,uint32_t* counts,double* objective,unsigned blocks,unsigned k,unsigned d){
    const unsigned j=blockIdx.x*blockDim.x+threadIdx.x,q=k*d+k;
    if(j>q)return;
    double total=0;for(unsigned b=0;b<blocks;++b)total+=partial[size_t(b)*(q+1)+j];
    if(j<k*d)sums[j]=total;else if(j<q)counts[j-k*d]=uint32_t(total);else *objective=total;
}
struct Device {
    double *points=nullptr,*centers=nullptr,*sums=nullptr,*objective=nullptr;
    uint32_t *labels=nullptr,*counts=nullptr;
    double *partial=nullptr,*transfer=nullptr;
    uint32_t *host_labels=nullptr;
    cudaStream_t stream=nullptr;
    size_t aggregate_bytes=0;
    ~Device() {
        if(stream)cudaStreamDestroy(stream);
        if(points)cudaFree(points); if(centers)cudaFree(centers); if(sums)cudaFree(sums);
        if(labels)cudaFree(labels);
        if(partial)cudaFree(partial);if(transfer)cudaFreeHost(transfer);if(host_labels)cudaFreeHost(host_labels);
    }
    void allocate(unsigned n,unsigned k,unsigned d) {
        check(cudaMalloc(&points,size_t(n)*d*8));check(cudaMalloc(&centers,k*d*sizeof(double)));
        const size_t value_count=k*d+1;  // Sums, objective, then integer counts with no padding.
        aggregate_bytes=value_count*sizeof(double)+k*sizeof(uint32_t);
        check(cudaMalloc(&sums,aggregate_bytes));
        objective=sums+k*d;
        counts=reinterpret_cast<uint32_t*>(sums+value_count);
        check(cudaMalloc(&labels,n*4));
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaMalloc(&partial,size_t((n+255)/256)*(k*d+k+1)*8));
        check(cudaMallocHost(&transfer,k*d*sizeof(double)+aggregate_bytes));check(cudaMallocHost(&host_labels,n*4));
    }
};
struct PinnedPoints {
    double* data=nullptr;
    ~PinnedPoints(){if(data)cudaFreeHost(data);}
};

int main(int argc,char** argv) {
    try {
        if(argc<6)throw std::runtime_error("Usage: atlas_kmeans INPUT CPU_COUNTS MAX_ITERS WORKERS fixed|converge [feedback] [verify|timing]");
        bool feedback=false,timing=false,reduced=false,streamed=false,gpu_first=false,armed=false;
        unsigned batch=1,phase_at=100001,phase_workers=1;
        for(int i=6;i<argc;++i){
            const std::string option=argv[i];
            if(option=="feedback")feedback=true;
            else if(option=="timing")timing=true;
            else if(option=="reduced")reduced=true;
            else if(option=="streamed")streamed=true;
            else if(option=="gpu-first")gpu_first=true;
            else if(option=="armed")armed=true;
            else if(option.rfind("batch=",0)==0)batch=number(option.substr(6),1,1000);
            else if(option.rfind("phase=",0)==0){auto colon=option.find(':');if(colon==std::string::npos)throw std::runtime_error("phase=ITER:WORKERS required");
                phase_at=number(option.substr(6,colon-6),1,100000);phase_workers=number(option.substr(colon+1),1,8);}
            else if(option!="verify")throw std::runtime_error("Unknown execution option");
        }
        const int64_t preparation_start=km::now();
        const unsigned max_iters=number(argv[3],1,100000),workers=number(argv[4],1,8);
        if(phase_workers>workers||(!feedback&&batch!=1))throw std::runtime_error("Invalid phase workers or batch without feedback");
        const std::string mode=argv[5]; if(mode!="fixed"&&mode!="converge")throw std::runtime_error("Unknown execution mode");
        std::ifstream input(argv[1],std::ios::binary);
        char magic[8]; uint32_t n=0,d=0,k=0;
        input.read(magic,8);input.read(reinterpret_cast<char*>(&n),4);input.read(reinterpret_cast<char*>(&d),4);input.read(reinterpret_cast<char*>(&k),4);
        if(!input||std::string(magic,8)!="ATLSKM01"||n<1||n>262147||d<1||d>32||k<1||k>32||k>n)throw std::runtime_error("Invalid dataset header");
        std::vector<unsigned> schedule;
        std::stringstream shares(argv[2]);std::string token;
        while(std::getline(shares,token,','))schedule.push_back(number(token,0,n));
        if(schedule.size()!=1&&schedule.size()!=max_iters)throw std::runtime_error("Schedule must have one count or MAX_ITERS counts");
        if(feedback&&schedule.size()!=1)throw std::runtime_error("Feedback requires one initial CPU count");
        int gpu_count=0;check(cudaGetDeviceCount(&gpu_count));if(gpu_count!=1)throw std::runtime_error("Exactly one GPU required");
        char pci[32];check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));
        if(armed&&!feedback)throw std::runtime_error("Armed execution requires barrier feedback");
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));if(prop.major<6)throw std::runtime_error("FP64 atomics require compute capability >=6.0");
        PinnedPoints points;check(cudaMallocHost(&points.data,size_t(n)*d*8));
        std::vector<double> initial(k*d),centers(k*d),gpu_sums(k*d),sums(k*d);
        input.read(reinterpret_cast<char*>(points.data),size_t(n)*d*8);
        input.read(reinterpret_cast<char*>(initial.data()),k*d*8);
        if(!input||input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Truncated dataset or trailing bytes");
        for(size_t i=0;i<size_t(n)*d;++i)if(!std::isfinite(points.data[i])||std::abs(points.data[i])>1e6)throw std::runtime_error("Invalid point coordinate");
        for(double v:initial)if(!std::isfinite(v)||std::abs(v)>1e6)throw std::runtime_error("Invalid initial centroid");
        centers=initial;
        std::vector<uint32_t> labels(n,0xffffffffU),previous(n,0xffffffffU),gpu_counts(k),counts(k);
        Device device;device.allocate(n,k,d);
        km::Cpu cpu(k,d,workers); unsigned cpu_items=schedule[0];
        Worker worker([&]{cpu.assign(points.data,centers.data(),labels.data(),cpu_items);});
        km::Reference reference(n,k,d,initial);
        unsigned resident_begin=n,previous_cut=n,executed=0;
        int64_t active_ns=0,first_release=0,last_end=0,control_begin=0,total_control_ns=0;
        bool converged=false;
        double final_quality=0;
        int64_t validation_total=0;
        std::cout<<std::setprecision(17);
        std::cout<<"{\"kind\":\"configuration\",\"schema_version\":\"4.1\",\"validation_mode\":\""<<(timing?"timing":"verify")<<"\",\"clock\":\"CLOCK_MONOTONIC\",\"n\":"<<n<<",\"d\":"<<d<<",\"k\":"<<k
                 <<",\"max_iterations\":"<<max_iters<<",\"workers_requested\":"<<workers<<",\"mode\":\""<<mode<<"\",\"atol\":1e-8,\"rtol\":1e-10,\"shift_tolerance\":1e-6,\"preparation_start_ns\":"<<preparation_start
                 <<",\"pci_bus_id\":\""<<pci<<"\",\"armed\":"<<(armed?"true":"false")
                 <<",\"dispatch_order\":\""<<(gpu_first?"gpu_first":"cpu_first")<<"\""
                 <<",\"feedback\":"<<(feedback?"true":"false")<<",\"feedback_batch\":"<<batch
                 <<",\"reduction\":\""<<(reduced?"hierarchical":"atomic")<<"\",\"transfer_mode\":\""<<(streamed?"streamed":"diagnostic")<<"\",\"initial_centroids\":";array_json(initial);std::cout<<"}"<<std::endl;
        if(armed){std::string ready;if(!std::getline(std::cin,ready)||ready!="start")throw std::runtime_error("Missing armed start acknowledgement");}
        for(int iteration=-1;iteration<int(max_iters);++iteration) {
            const bool warmup=iteration<0;
            cpu.workers=!warmup&&unsigned(iteration)>=phase_at?phase_workers:workers;
            int64_t control_ns=0;
            if(feedback&&iteration>0) {
                if(unsigned(iteration)%batch==0){
                std::string request;
                if(!std::getline(std::cin,request))throw std::runtime_error("Feedback input closed before next partition");
                cpu_items=number(request,0,n);
                }
                control_ns=km::now()-control_begin;total_control_ns+=control_ns;
            } else cpu_items=schedule.size()==1?schedule[0]:schedule[warmup?0:iteration];
            const unsigned gpu_items=n-cpu_items,resident_before=resident_begin,old_cut=previous_cut;
            std::fill(labels.begin(),labels.end(),0xffffffffU);
            const int64_t release=km::now(); if(!warmup&&!executed)first_release=release;
            nvtxRangePushA(warmup?"kmeans_warmup":"kmeans_iteration");
            if(cpu_items&&(!gpu_first||!gpu_items))worker.release();
            const int64_t gpu_dispatch=km::now();
            int64_t h2d_end=gpu_dispatch,compute_end=gpu_dispatch,gpu_end=gpu_dispatch;
            uint64_t point_bytes=0;
            double gpu_objective=0;
            std::fill(gpu_counts.begin(),gpu_counts.end(),0);std::fill(gpu_sums.begin(),gpu_sums.end(),0);
            if(gpu_items) {
                if(cpu_items<resident_begin) {
                    point_bytes=uint64_t(resident_begin-cpu_items)*d*8;
                    check(cudaMemcpyAsync(device.points+size_t(cpu_items)*d,points.data+size_t(cpu_items)*d,point_bytes,cudaMemcpyHostToDevice,device.stream));
                    resident_begin=cpu_items;
                }
                std::copy(centers.begin(),centers.end(),device.transfer);
                check(cudaMemcpyAsync(device.centers,device.transfer,k*d*8,cudaMemcpyHostToDevice,device.stream));
                // The hierarchical finalizer overwrites every aggregate, including empty clusters.
                if(!reduced)check(cudaMemsetAsync(device.sums,0,device.aggregate_bytes,device.stream));
                check(cudaMemsetAsync(device.labels+cpu_items,0xff,gpu_items*4,device.stream));
                if(!streamed){check(cudaStreamSynchronize(device.stream));h2d_end=km::now();}
                const unsigned blocks=(gpu_items+255)/256;
                if(reduced){
                    kmeans_assign_reduced<<<blocks,256,(k*d+k+256)*8,device.stream>>>(device.points,device.centers,device.labels,device.partial,cpu_items,n,k,d);
                    check(cudaGetLastError());
                    kmeans_finalize<<<(k*d+k+1+255)/256,256,0,device.stream>>>(device.partial,device.sums,device.counts,device.objective,blocks,k,d);
                }else kmeans_assign<<<blocks,256,0,device.stream>>>(device.points,device.centers,device.labels,device.sums,device.counts,device.objective,cpu_items,n,k,d);
                // Release CPU work immediately after submitting real GPU work.
                // No sleep, extra arithmetic, or artificial device waiting.
                if(cpu_items&&gpu_first)worker.release();
                check(cudaGetLastError());if(!streamed){check(cudaStreamSynchronize(device.stream));compute_end=km::now();}
                double* host_sums=device.transfer+k*d;
                double* host_objective=host_sums+k*d;
                auto* host_counts=reinterpret_cast<uint32_t*>(host_objective+1);
                check(cudaMemcpyAsync(device.host_labels+cpu_items,device.labels+cpu_items,gpu_items*4,cudaMemcpyDeviceToHost,device.stream));
                check(cudaMemcpyAsync(host_sums,device.sums,device.aggregate_bytes,cudaMemcpyDeviceToHost,device.stream));
                check(cudaStreamSynchronize(device.stream));
                std::copy(device.host_labels+cpu_items,device.host_labels+n,labels.begin()+cpu_items);
                std::copy(host_sums,host_sums+k*d,gpu_sums.begin());std::copy(host_counts,host_counts+k,gpu_counts.begin());gpu_objective=*host_objective;
                gpu_end=km::now();
            }
            if(cpu_items)worker.wait();
            const int64_t barrier_end=km::now();
            double shift=0; unsigned empty=0,changed=0;
            for(unsigned c=0;c<k;++c) {
                counts[c]=(cpu_items?cpu.result.counts[c]:0)+gpu_counts[c];
                double square_shift=0;
                if(!counts[c])++empty;
                for(unsigned j=0;j<d;++j) {
                    const unsigned q=c*d+j;
                    sums[q]=(cpu_items?cpu.result.sums[q]:0)+gpu_sums[q];
                    const double updated=counts[c]?sums[q]/counts[c]:centers[q];
                    const double delta=updated-centers[q];square_shift+=delta*delta;centers[q]=updated;
                }
                shift=std::max(shift,std::sqrt(square_shift));
            }
            for(unsigned i=0;i<n;++i)changed+=(labels[i]!=previous[i]);
            std::copy(labels.begin(),labels.end(),previous.begin());
            const double objective=(cpu_items?cpu.result.objective:0)+gpu_objective;
            converged=changed==0||shift<=1e-6;
            const int64_t end=km::now();nvtxRangePop();
            // Validation follows the complete iteration; it never drives the backend's state.
            const bool checked=!timing||warmup;
            if(checked)reference.step(points.data,n,k,d);
            unsigned label_errors=0,count_errors=0,numeric_errors=0;
            long double quality=0;
            for(unsigned i=0;i<n;++i) {
                if(checked)label_errors+=(labels[i]!=reference.labels[i]);
                if(labels[i]>=k) {++numeric_errors;continue;}
                for(unsigned j=0;j<d;++j) {
                    long double delta=points.data[size_t(i)*d+j]-centers[labels[i]*d+j];quality+=delta*delta;
                }
            }
            if(checked)for(unsigned c=0;c<k;++c) count_errors+=(counts[c]!=reference.counts[c]);
            for(unsigned q=0;q<k*d;++q) {
                numeric_errors+=!std::isfinite(sums[q])||!std::isfinite(centers[q]);
                if(checked){numeric_errors+=!km::close(sums[q],reference.sums[q]);
                    numeric_errors+=!km::close(centers[q],reference.centers[q]);}
            }
            numeric_errors+=!std::isfinite(objective);
            if(checked)numeric_errors+=!km::close(objective,reference.objective);
            if(static_cast<double>(quality)>objective&&!km::close(static_cast<double>(quality),objective))++numeric_errors;
            const bool passed=!label_errors&&!count_errors&&!numeric_errors;
            const int64_t validation_end=km::now();
            if(!warmup)validation_total+=validation_end-end;
            // Include previous-record export, controller/IPC, and partition parsing
            // in the next measured iteration's control interval, outside validation.
            if(feedback&&!warmup)control_begin=validation_end;
            const bool final=!warmup&&(iteration+1==int(max_iters)||(mode=="converge"&&converged));
            const bool due=!warmup&&!final&&(unsigned(iteration+1)%batch==0);
            std::cout<<"{\"kind\":\"iteration\",\"index\":"<<iteration<<",\"phase\":"<<(!warmup&&unsigned(iteration)>=phase_at?1:0)
                     <<",\"feedback_due\":"<<(due?"true":"false")<<",\"warmup\":"<<(warmup?"true":"false")<<",\"status\":\""<<(passed?"passed":"failed")
                     <<"\",\"validation\":\""<<(checked?"checked":"deferred")<<"\",\"cpu_items\":"<<cpu_items<<",\"gpu_items\":"<<gpu_items<<",\"resident_before\":"<<resident_before<<",\"resident_after\":"<<resident_begin
                     <<",\"ownership_changed_points\":"<<(cpu_items>old_cut?cpu_items-old_cut:old_cut-cpu_items)<<",\"h2d_point_bytes\":"<<point_bytes
                     <<",\"h2d_centroid_bytes\":"<<(gpu_items?k*d*8:0)<<",\"d2h_label_bytes\":"<<uint64_t(gpu_items)*4<<",\"d2h_aggregate_bytes\":"<<(gpu_items?k*d*8+k*4+8:0)
                     <<",\"release_ns\":"<<release<<",\"cpu_start_ns\":"<<(cpu_items?std::to_string(cpu.result.start):"null")<<",\"cpu_end_ns\":"<<(cpu_items?std::to_string(cpu.result.end):"null")
                     <<",\"gpu_dispatch_ns\":"<<(gpu_items?std::to_string(gpu_dispatch):"null")<<",\"h2d_end_ns\":"<<(gpu_items&&!streamed?std::to_string(h2d_end):"null")
                     <<",\"compute_end_ns\":"<<(gpu_items&&!streamed?std::to_string(compute_end):"null")<<",\"gpu_end_ns\":"<<(gpu_items?std::to_string(gpu_end):"null")
                     <<",\"barrier_end_ns\":"<<barrier_end<<",\"end_ns\":"<<end<<",\"validation_end_ns\":"<<validation_end
                     <<",\"control_ns\":"<<control_ns<<",\"workers_used\":"<<(cpu_items?cpu.result.workers:0)<<",\"objective_before_update\":"<<objective<<",\"objective_after_update\":"<<static_cast<double>(quality)
                     <<",\"max_centroid_shift\":"<<shift<<",\"changed_labels\":"<<changed<<",\"empty_clusters\":"<<empty<<",\"converged\":"<<(converged?"true":"false")
                     <<",\"label_errors\":"<<label_errors<<",\"count_errors\":"<<count_errors<<",\"numeric_errors\":"<<numeric_errors<<",\"label_hash\":\""<<label_hash(labels)<<"\",\"counts\":";
            array_json(counts);std::cout<<",\"sums\":";array_json(sums);std::cout<<",\"centroids\":";array_json(centers);
            if(n<=513){std::cout<<",\"labels\":";array_json(labels);}
            std::cout<<"}\n";if(warmup||due||final)std::cout.flush();
            if(!passed)return 1;
            if(warmup) {
                centers=initial;reference=km::Reference(n,k,d,initial);resident_begin=n;previous_cut=n;
                std::fill(previous.begin(),previous.end(),0xffffffffU);
            } else {
                ++executed;active_ns+=end-release;last_end=end;final_quality=static_cast<double>(quality);previous_cut=cpu_items;
                if(mode=="converge"&&converged)break;
            }
        }
        const int64_t final_validation_start=km::now();
        if(timing){
            for(unsigned i=0;i<executed;++i)reference.step_parallel(points.data,n,k,d,workers);
            if(labels!=reference.labels||counts!=reference.counts)throw std::runtime_error("Deferred membership/count validation failed");
            for(unsigned q=0;q<k*d;++q)if(!km::close(centers[q],reference.centers[q])||!km::close(sums[q],reference.sums[q]))
                throw std::runtime_error("Deferred centroid validation failed");
        }
        const int64_t final_validation_ns=km::now()-final_validation_start;
        std::cout<<"{\"kind\":\"summary\",\"status\":\"passed\",\"independent_validation\":\"passed\",\"validation_ns\":"<<validation_total
                 <<",\"final_validation_ns\":"<<final_validation_ns<<",\"final_reference_workers_requested\":"<<(timing&&n>=1024?workers:1)
                 <<",\"iterations\":"<<executed<<",\"active_ns\":"<<active_ns<<",\"loop_elapsed_ns\":"<<last_end-first_release
                 <<",\"control_ns\":"<<total_control_ns<<",\"active_and_control_ns\":"<<active_ns+total_control_ns
                 <<",\"final_objective\":"<<final_quality<<",\"stop_reason\":\""<<(mode=="fixed"?"fixed_budget":converged?"converged":"max_iterations")<<"\",\"final_labels\":";
        array_json(labels);std::cout<<",\"final_centroids\":";array_json(centers);std::cout<<"}"<<std::endl;
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
