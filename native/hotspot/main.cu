// Thermal equations/constants traced to Rodinia CUDA; see references/rodinia/LICENSE.
#include "backend.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
static unsigned number(const std::string& s,unsigned lo,unsigned hi){
    if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("Expected unsigned integer");
    auto v=std::stoull(s);if(v<lo||v>hi)throw std::runtime_error("Argument outside supported range");return unsigned(v);
}
class Worker {
    std::mutex mutex;std::condition_variable cv;bool pending=false,done=false,stop=false;
    std::function<void()> task;std::exception_ptr failure;std::thread thread;
    void loop(){std::unique_lock<std::mutex> lock(mutex);for(;;){
        cv.wait(lock,[&]{return pending||stop;});if(stop)return;pending=false;lock.unlock();
        try{task();}catch(...){failure=std::current_exception();}
        lock.lock();done=true;cv.notify_all();
    }}
public:
    explicit Worker(std::function<void()> fn):task(std::move(fn)),thread(&Worker::loop,this){}
    ~Worker(){{std::lock_guard<std::mutex> lock(mutex);stop=true;cv.notify_all();}thread.join();}
    void release(){std::lock_guard<std::mutex> lock(mutex);failure=nullptr;done=false;pending=true;cv.notify_all();}
    void wait(){std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return done;});if(failure)std::rethrow_exception(failure);}
};
struct Storage {
    double *ha=nullptr,*hb=nullptr,*power=nullptr,*da=nullptr,*db=nullptr,*dp=nullptr;
    cudaStream_t stream=nullptr;
    ~Storage(){if(stream)cudaStreamDestroy(stream);if(da)cudaFree(da);if(db)cudaFree(db);if(dp)cudaFree(dp);
        if(ha)cudaFreeHost(ha);if(hb)cudaFreeHost(hb);if(power)cudaFreeHost(power);}
    void allocate(size_t bytes){check(cudaMallocHost(&ha,bytes));check(cudaMallocHost(&hb,bytes));check(cudaMallocHost(&power,bytes));
        check(cudaMalloc(&da,bytes));check(cudaMalloc(&db,bytes));check(cudaMalloc(&dp,bytes));
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
};

__global__ void hotspot_step(const double* old,double* next,const double* power,unsigned rows,unsigned cols,unsigned cut,
                              double gain,double wx,double wy,double wz){
    size_t i=size_t(cut)*cols+size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*cols)return;
    unsigned r=i/cols,c=i%cols;double t=old[i];
    double north=old[(r?r-1:r)*cols+c],south=old[(r+1<rows?r+1:r)*cols+c];
    double west=old[size_t(r)*cols+(c?c-1:c)],east=old[size_t(r)*cols+(c+1<cols?c+1:c)];
    next[i]=t+gain*(power[i]+(south+north-2*t)*wy+(east+west-2*t)*wx+(80.-t)*wz);
}
static void array_json(const std::vector<double>& values){std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
static unsigned validate(const std::vector<double>& actual,const std::vector<long double>& reference,double& maximum){
    unsigned errors=0;maximum=0;
    for(size_t i=0;i<actual.size();++i){
        double error=static_cast<double>(std::abs(static_cast<long double>(actual[i])-reference[i]));
        maximum=std::max(maximum,error);
        if(!std::isfinite(actual[i])||error>1e-8+1e-10*std::abs(reference[i]))++errors;
    }return errors;
}
static void output(const char* path,unsigned rows,unsigned cols,const std::vector<double>& values){
    FILE* f=std::fopen(path,"wbx");if(!f)throw std::runtime_error("Cannot create new temperature output");
    uint32_t r=rows,c=cols;
    bool okay=std::fwrite("ATLSTP01",1,8,f)==8&&std::fwrite(&r,4,1,f)==1&&std::fwrite(&c,4,1,f)==1
        &&std::fwrite(values.data(),8,values.size(),f)==values.size();
    if(std::fclose(f)||!okay)throw std::runtime_error("Temperature output write failed");
}

int main(int argc,char** argv){try{
    if(argc<7)throw std::runtime_error("Usage: atlas_hotspot INPUT CPU_ROWS_COMMA STEPS WORKERS verify|timing OUTPUT [feedback] [batch=N] [phase=ITER:WORKERS]");
    bool feedback=false,overlap=false,gpu_first=false,armed=false;unsigned batch=1,phase_at=100001,phase_workers=1;
    for(int i=7;i<argc;++i){const std::string option=argv[i];
        if(option=="feedback")feedback=true;
        else if(option=="overlap")overlap=true;
        else if(option=="gpu-first")gpu_first=true;
        else if(option=="armed")armed=true;
        else if(option.rfind("batch=",0)==0)batch=number(option.substr(6),1,1000);
        else if(option.rfind("phase=",0)==0){auto colon=option.find(':');if(colon==std::string::npos)throw std::runtime_error("phase=ITER:WORKERS required");
            phase_at=number(option.substr(6,colon-6),1,100000);phase_workers=number(option.substr(colon+1),1,8);}
        else throw std::runtime_error("Unknown feedback option");}
    const unsigned steps=number(argv[3],1,100000),initial_workers=number(argv[4],1,8);
    if(gpu_first&&overlap)throw std::runtime_error("gpu-first and halo-overlap are separate dispatch experiments");
    unsigned workers=initial_workers;
    if(phase_workers>workers||(!feedback&&batch!=1))throw std::runtime_error("Invalid phase workers or batch without feedback");
    const std::string mode=argv[5];if(mode!="verify"&&mode!="timing")throw std::runtime_error("Unknown validation mode");
    std::ifstream input(argv[1],std::ios::binary);char magic[8];uint32_t rows=0,cols=0;
    input.read(magic,8);input.read(reinterpret_cast<char*>(&rows),4);input.read(reinterpret_cast<char*>(&cols),4);
    if(!input||std::string(magic,8)!="ATLSHS01"||rows<1||rows>2048||cols<1||cols>2048)throw std::runtime_error("Invalid input header");
    std::vector<unsigned> schedule;std::stringstream parts(argv[2]);std::string part;
    while(std::getline(parts,part,','))schedule.push_back(number(part,0,rows));
    if((schedule.size()!=1&&schedule.size()!=steps)||(feedback&&schedule.size()!=1))throw std::runtime_error("Invalid partition schedule");
    int gpu_count=0;check(cudaGetDeviceCount(&gpu_count));if(gpu_count!=1)throw std::runtime_error("Exactly one GPU required");
    char pci[32];check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));
    if(armed&&!feedback)throw std::runtime_error("Armed execution requires barrier feedback");
    const size_t n=size_t(rows)*cols,bytes=n*8,row_bytes=cols*8;
    Storage storage;storage.allocate(bytes);std::vector<double> initial(n),snapshot(n);
    input.read(reinterpret_cast<char*>(initial.data()),bytes);input.read(reinterpret_cast<char*>(storage.power),bytes);
    if(!input||input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Truncated input or trailing bytes");
    const double max_power=3e6*.016*.016/n;
    for(size_t i=0;i<n;++i)if(!std::isfinite(initial[i])||initial[i]<0||initial[i]>500||!std::isfinite(storage.power[i])||storage.power[i]<0||storage.power[i]>max_power)
        throw std::runtime_error("Invalid temperature or cell power");
    const hs::Model m=hs::model(rows,cols);
    std::vector<long double> reference(initial.begin(),initial.end()),reference_next(n);
    double *host=storage.ha,*next_host=storage.hb,*device=storage.da,*next_device=storage.db;
    auto reset=[&]{std::copy(initial.begin(),initial.end(),host);std::fill(next_host,next_host+n,std::numeric_limits<double>::quiet_NaN());
        check(cudaMemsetAsync(device,0xff,bytes,storage.stream));check(cudaMemsetAsync(next_device,0xff,bytes,storage.stream));check(cudaStreamSynchronize(storage.stream));
        reference.assign(initial.begin(),initial.end());};
    reset();unsigned cut=schedule[0],old_cut=rows,power_begin=rows;
    int64_t cpu_begin=0,cpu_end=0;unsigned interior_cut=cut;
    Worker worker([&]{hs::cpu_step(host,next_host,storage.power,rows,cols,interior_cut,workers,m,cpu_begin,cpu_end);});
    int64_t active=0,control_total=0,control_begin=0,first_release=0,last_end=0;
    uint64_t validation_bytes=0;double final_error=0;
    std::cout<<std::setprecision(17);
    std::cout<<"{\"kind\":\"configuration\",\"schema_version\":\"6.2\",\"communication_mode\":\""<<(overlap?"overlap":"serial")<<"\",\"feedback_batch\":"<<batch<<",\"clock\":\"CLOCK_MONOTONIC\",\"rows\":"<<rows<<",\"cols\":"<<cols
        <<",\"pci_bus_id\":\""<<pci<<"\",\"armed\":"<<(armed?"true":"false")
        <<",\"dispatch_order\":\""<<(gpu_first?"gpu_first":"cpu_first")<<"\""
        <<",\"steps\":"<<steps<<",\"workers\":"<<workers<<",\"mode\":\""<<mode<<"\",\"feedback\":"<<(feedback?"true":"false")
        <<",\"atol\":1e-8,\"rtol\":1e-10,\"cap\":"<<m.cap<<",\"rx\":"<<m.rx<<",\"ry\":"<<m.ry<<",\"rz\":"<<m.rz<<",\"dt\":"<<m.dt
        <<",\"nominal_dt\":"<<m.nominal<<",\"stability_sum\":"<<m.stability<<"}"<<std::endl;
    if(armed){std::string ready;if(!std::getline(std::cin,ready)||ready!="start")throw std::runtime_error("Missing armed start acknowledgement");}
    for(int iteration=-1;iteration<int(steps);++iteration){
        bool warmup=iteration<0;int64_t control=0;
        workers=!warmup&&unsigned(iteration)>=phase_at?phase_workers:initial_workers;
        if(feedback&&iteration>0){
            if(unsigned(iteration)%batch==0){std::string request;if(!std::getline(std::cin,request))throw std::runtime_error("Feedback input closed");cut=number(request,0,rows);}
            control=hs::now()-control_begin;control_total+=control;
        }else cut=schedule[schedule.size()==1||warmup?0:iteration];
        const bool initial_step=warmup||iteration==0;
        uint64_t init_h2d=initial_step?(rows-cut)*row_bytes:0;
        uint64_t migrate_h2d=!initial_step&&cut<old_cut?(old_cut-cut)*row_bytes:0;
        uint64_t migrate_d2h=!initial_step&&cut>old_cut?(cut-old_cut)*row_bytes:0;
        uint64_t halo=0<cut&&cut<rows?row_bytes:0,power_bytes=cut<power_begin?(power_begin-cut)*row_bytes:0;
        const int64_t release=hs::now();if(iteration==0)first_release=release;
        nvtxRangePushA(warmup?"hotspot_warmup":"hotspot_iteration");
        if(init_h2d||migrate_h2d)check(cudaMemcpyAsync(device+size_t(cut)*cols,host+size_t(cut)*cols,init_h2d+migrate_h2d,cudaMemcpyHostToDevice,storage.stream));
        if(migrate_d2h)check(cudaMemcpyAsync(host+size_t(old_cut)*cols,device+size_t(old_cut)*cols,migrate_d2h,cudaMemcpyDeviceToHost,storage.stream));
        if(power_bytes)check(cudaMemcpyAsync(storage.dp+size_t(cut)*cols,storage.power+size_t(cut)*cols,power_bytes,cudaMemcpyHostToDevice,storage.stream));
        const bool split_boundary=overlap&&halo;
        // The pinned D2H migration and halo H2D are ordered in one stream.
        // Fence here only when there is no halo fence, or the CPU will read migrated rows early.
        if((!halo&&(init_h2d||migrate_h2d||migrate_d2h||power_bytes))||(split_boundary&&migrate_d2h))
            check(cudaStreamSynchronize(storage.stream));
        interior_cut=split_boundary?cut-1:cut;
        const int64_t migration_end=hs::now();
        if(split_boundary&&interior_cut)worker.release();
        if(halo){
            check(cudaMemcpyAsync(device+size_t(cut-1)*cols,host+size_t(cut-1)*cols,halo,cudaMemcpyHostToDevice,storage.stream));
            check(cudaMemcpyAsync(host+size_t(cut)*cols,device+size_t(cut)*cols,halo,cudaMemcpyDeviceToHost,storage.stream));
            check(cudaStreamSynchronize(storage.stream));
        }
        const int64_t communication_end=hs::now();
        if(!split_boundary&&interior_cut&&!gpu_first)worker.release();
        const int64_t compute_release=split_boundary?migration_end:communication_end;
        int64_t gpu_end=0;
        if(cut<rows){
            hotspot_step<<<((rows-cut)*cols+255)/256,256,0,storage.stream>>>(device,next_device,storage.dp,rows,cols,cut,m.dt/m.cap,1/m.rx,1/m.ry,1/m.rz);
            check(cudaGetLastError());
        }
        if(gpu_first&&interior_cut)worker.release();
        int64_t boundary_begin=0,boundary_end=0;
        if(split_boundary)hs::cpu_step(host,next_host,storage.power,rows,cols,cut,1,m,boundary_begin,boundary_end,cut-1);
        if(cut<rows){check(cudaStreamSynchronize(storage.stream));gpu_end=hs::now();}
        if(interior_cut)worker.wait();
        if(split_boundary){cpu_begin=interior_cut?std::min(cpu_begin,boundary_begin):boundary_begin;cpu_end=std::max(interior_cut?cpu_end:boundary_end,boundary_end);}
        std::swap(host,next_host);std::swap(device,next_device);
        const int64_t end=hs::now();nvtxRangePop();
        unsigned errors=0;double max_error=0;uint64_t checked_bytes=0;
        if(mode=="verify"||warmup){
            std::copy(host,host+size_t(cut)*cols,snapshot.begin());
            if(cut<rows){checked_bytes=(rows-cut)*row_bytes;check(cudaMemcpy(snapshot.data()+size_t(cut)*cols,device+size_t(cut)*cols,checked_bytes,cudaMemcpyDeviceToHost));}
            hs::reference_step(reference,reference_next,storage.power,rows,cols,m);
            errors=validate(snapshot,reference,max_error);
            if(!warmup)validation_bytes+=checked_bytes;
        }
        const int64_t validation_end=hs::now();
        if(feedback&&!warmup)control_begin=validation_end;
        const bool final=!warmup&&iteration+1==int(steps),due=!warmup&&!final&&(unsigned(iteration+1)%batch==0);
        std::cout<<"{\"kind\":\"iteration\",\"index\":"<<iteration<<",\"phase\":"<<(!warmup&&unsigned(iteration)>=phase_at?1:0)
            <<",\"workers_used\":"<<workers<<",\"feedback_due\":"<<(due?"true":"false")<<",\"warmup\":"<<(warmup?"true":"false")<<",\"cpu_rows\":"<<cut
            <<",\"old_cpu_rows\":"<<old_cut<<",\"input_epoch\":"<<(warmup?0:iteration)<<",\"output_epoch\":"<<(warmup?1:iteration+1)
            <<",\"status\":\""<<(errors?"failed":"completed")<<"\",\"validation\":\""<<(mode=="verify"||warmup?"checked":"deferred")
            <<"\",\"errors\":"<<errors<<",\"max_error\":"<<max_error<<",\"initial_h2d_bytes\":"<<init_h2d<<",\"migration_h2d_bytes\":"<<migrate_h2d
            <<",\"migration_d2h_bytes\":"<<migrate_d2h<<",\"halo_h2d_bytes\":"<<halo<<",\"halo_d2h_bytes\":"<<halo<<",\"power_h2d_bytes\":"<<power_bytes
            <<",\"validation_d2h_bytes\":"<<checked_bytes<<",\"power_begin_after\":"<<std::min(power_begin,cut)<<",\"control_ns\":"<<control
            <<",\"release_ns\":"<<release<<",\"compute_release_ns\":"<<compute_release<<",\"communication_end_ns\":"<<communication_end<<",\"cpu_start_ns\":"<<(cut?std::to_string(cpu_begin):"null")
            <<",\"cpu_end_ns\":"<<(cut?std::to_string(cpu_end):"null")<<",\"gpu_end_ns\":"<<(cut<rows?std::to_string(gpu_end):"null")
            <<",\"end_ns\":"<<end<<",\"validation_end_ns\":"<<validation_end;
        if((mode=="verify"||warmup)&&n<=1024){std::cout<<",\"temperatures\":";array_json(snapshot);}
        std::cout<<"}\n";if(warmup||due||final)std::cout.flush();
        if(errors)return 1;
        if(warmup){reset();old_cut=rows;power_begin=rows;}
        else{active+=end-release;last_end=end;old_cut=cut;power_begin=std::min(power_begin,cut);final_error=max_error;}
    }
    const int64_t gather_begin=hs::now();uint64_t gather_bytes=0;
    if(mode=="timing"){
        std::copy(host,host+size_t(cut)*cols,snapshot.begin());
        if(cut<rows){gather_bytes=(rows-cut)*row_bytes;check(cudaMemcpy(snapshot.data()+size_t(cut)*cols,device+size_t(cut)*cols,gather_bytes,cudaMemcpyDeviceToHost));}
    }
    const int64_t gather_end=hs::now();
    if(mode=="timing")for(unsigned i=0;i<steps;++i)hs::reference_step(reference,reference_next,storage.power,rows,cols,m,n>=1024?initial_workers:1);
    unsigned errors=validate(snapshot,reference,final_error);
    if(errors)throw std::runtime_error("Final independent reference validation failed");
    output(argv[6],rows,cols,snapshot);
    std::cout<<"{\"kind\":\"summary\",\"status\":\"passed\",\"steps\":"<<steps<<",\"errors\":0,\"max_error\":"<<final_error
        <<",\"final_reference_workers_requested\":"<<(mode=="timing"&&n>=1024?initial_workers:1)
        <<",\"active_ns\":"<<active<<",\"control_ns\":"<<control_total<<",\"gather_ns\":"<<gather_end-gather_begin<<",\"gather_d2h_bytes\":"<<gather_bytes
        <<",\"validation_d2h_bytes\":"<<validation_bytes<<",\"work_and_control_and_gather_ns\":"<<active+control_total+gather_end-gather_begin
        <<",\"loop_elapsed_ns\":"<<last_end-first_release<<",\"simulated_seconds\":"<<steps*m.dt<<"}"<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
