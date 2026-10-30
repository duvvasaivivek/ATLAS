#include "backend.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <time.h>
#include <omp.h>
#include <nvtx3/nvToolsExt.h>

namespace km {
int64_t now() {
    timespec t{};
    if(clock_gettime(CLOCK_MONOTONIC,&t)) throw std::runtime_error("monotonic clock failed");
    return int64_t(t.tv_sec)*1000000000LL+t.tv_nsec;
}
bool close(double a,double b) {
    return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1e-8+1e-10*std::abs(b);
}
Cpu::Cpu(unsigned clusters,unsigned dimensions,unsigned team)
  :k(clusters),d(dimensions),workers(team),result(k,d) {
    for(unsigned i=0;i<workers;++i) scratch.emplace_back(k,d);
}
void Cpu::assign(const double* points,const double* centers,uint32_t* labels,unsigned count) {
    result.start=now(); result.objective=0; result.workers=0;
    std::fill(result.sums.begin(),result.sums.end(),0);
    std::fill(result.counts.begin(),result.counts.end(),0);
    for(auto& s:scratch) {
        std::fill(s.sums.begin(),s.sums.end(),0);
        std::fill(s.counts.begin(),s.counts.end(),0); s.objective=0;
    }
    if(count) {
        omp_set_dynamic(0);
        #pragma omp parallel num_threads(workers)
        {
            const unsigned id=omp_get_thread_num(),team=omp_get_num_threads();
            #pragma omp single
            result.workers=team;
            auto& s=scratch[id];
            const unsigned lo=uint64_t(count)*id/team,hi=uint64_t(count)*(id+1)/team;
            if(lo<hi) {
                nvtxRangePushA("kmeans_cpu_compute");
                for(unsigned i=lo;i<hi;++i) {
                    double best=std::numeric_limits<double>::infinity(); unsigned winner=0;
                    for(unsigned c=0;c<k;++c) {
                        double distance=0;
                        for(unsigned j=0;j<d;++j) {
                            double delta=points[size_t(i)*d+j]-centers[c*d+j];
                            distance+=delta*delta;
                        }
                        if(distance<best) {best=distance;winner=c;}
                    }
                    labels[i]=winner; ++s.counts[winner]; s.objective+=best;
                    for(unsigned j=0;j<d;++j) s.sums[winner*d+j]+=points[size_t(i)*d+j];
                }
                nvtxRangePop();
            }
        }
        for(auto& s:scratch) {
            for(unsigned c=0;c<k;++c) result.counts[c]+=s.counts[c];
            for(unsigned j=0;j<k*d;++j) result.sums[j]+=s.sums[j];
            result.objective+=s.objective;
        }
    }
    result.end=now();
}
void Reference::step(const double* points,unsigned n,unsigned k,unsigned d) {
    // Independent scalar assignment; long-double accumulation is the aggregate oracle.
    std::vector<long double> totals(k*d,0);
    long double cost=0;
    std::fill(counts.begin(),counts.end(),0);
    for(unsigned i=0;i<n;++i) {
        unsigned selected=0;
        double minimum=std::numeric_limits<double>::infinity();
        for(unsigned c=0;c<k;++c) {
            double square_sum=0;
            for(unsigned j=0;j<d;++j) {
                double difference=points[size_t(i)*d+j]-centers[c*d+j];
                square_sum=square_sum+difference*difference;
            }
            if(square_sum<minimum) {minimum=square_sum;selected=c;}
        }
        labels[i]=selected; ++counts[selected]; cost+=minimum;
        for(unsigned j=0;j<d;++j) totals[selected*d+j]+=points[size_t(i)*d+j];
    }
    for(unsigned c=0;c<k;++c) for(unsigned j=0;j<d;++j) {
        sums[c*d+j]=static_cast<double>(totals[c*d+j]);
        if(counts[c]) centers[c*d+j]=static_cast<double>(totals[c*d+j]/counts[c]);
    }
    objective=static_cast<double>(cost);
}
}
