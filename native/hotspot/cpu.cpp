#include "backend.hpp"
#include <algorithm>
#include <stdexcept>
#include <time.h>
#include <omp.h>
#include <nvtx3/nvToolsExt.h>

namespace hs {
int64_t now(){timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t))throw std::runtime_error("Clock failure");return int64_t(t.tv_sec)*1000000000LL+t.tv_nsec;}
Model model(unsigned rows,unsigned cols){
    const double dy=.016/rows,dx=.016/cols;
    Model m{};m.cap=.5*1.75e6*.0005*dx*dy;
    m.rx=dx/(2*100*.0005*dy);m.ry=dy/(2*100*.0005*dx);m.rz=.0005/(100*dy*dx);
    m.nominal=.001/(3e6/(.5*.0005*1.75e6));
    m.dt=std::min(m.nominal,.9*m.cap/(2/m.rx+2/m.ry+1/m.rz));
    m.stability=m.dt/m.cap*(2/m.rx+2/m.ry+1/m.rz);return m;
}
void cpu_step(const double* old,double* next,const double* power,unsigned rows,unsigned cols,
              unsigned cut,unsigned workers,const Model& m,int64_t& begin,int64_t& end,unsigned first_row){
    begin=now();omp_set_dynamic(0);
    const double gain=m.dt/m.cap,wx=1/m.rx,wy=1/m.ry,wz=1/m.rz;
    #pragma omp parallel num_threads(workers)
    {
        const unsigned id=omp_get_thread_num(),team=omp_get_num_threads();
        const unsigned lo=first_row+uint64_t(cut-first_row)*id/team,hi=first_row+uint64_t(cut-first_row)*(id+1)/team;
        if(lo<hi){
        nvtxRangePushA("hotspot_cpu_compute");
        for(unsigned r=lo;r<hi;++r)for(unsigned c=0;c<cols;++c){
            const size_t i=size_t(r)*cols+c;
            const double t=old[i],north=old[(r?r-1:r)*cols+c],south=old[(r+1<rows?r+1:r)*cols+c];
            const double west=old[size_t(r)*cols+(c?c-1:c)],east=old[size_t(r)*cols+(c+1<cols?c+1:c)];
            next[i]=t+gain*(power[i]+(south+north-2*t)*wy+(east+west-2*t)*wx+(m.ambient-t)*wz);
        }
        nvtxRangePop();
        }
        nvtxRangePushA("hotspot_cpu_wait");
        #pragma omp barrier
        nvtxRangePop();
    }
    end=now();
}
void reference_step(std::vector<long double>& old,std::vector<long double>& next,const double* power,
                    unsigned rows,unsigned cols,const Model& m,unsigned workers){
    // Independent cells can run in parallel without changing scalar flux order
    // or the long-double evolving state. The verification path remains serial.
    auto row=[&](unsigned r){for(unsigned c=0;c<cols;++c){
        const size_t i=size_t(r)*cols+c;const long double t=old[i];
        long double flux=power[i]+(m.ambient-t)/m.rz;
        if(r)flux+=(old[i-cols]-t)/m.ry;
        if(r+1<rows)flux+=(old[i+cols]-t)/m.ry;
        if(c)flux+=(old[i-1]-t)/m.rx;
        if(c+1<cols)flux+=(old[i+1]-t)/m.rx;
        next[i]=t+(static_cast<long double>(m.dt)/m.cap)*flux;
    }};
    if(workers>1){
        #pragma omp parallel for schedule(static) num_threads(workers)
        for(unsigned r=0;r<rows;++r)row(r);
    }else for(unsigned r=0;r<rows;++r)row(r);
    old.swap(next);
}
}
