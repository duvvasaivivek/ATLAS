#pragma once
#include <cstdint>
#include <vector>

namespace km {
int64_t now();
struct Stats {
    std::vector<double> sums;
    std::vector<uint32_t> counts;
    double objective=0;
    int workers=0;
    int64_t start=0,end=0;
    Stats(unsigned k,unsigned d):sums(k*d),counts(k) {}
};
struct Cpu {
    unsigned k,d,workers;
    std::vector<Stats> scratch;
    Stats result;
    Cpu(unsigned clusters,unsigned dimensions,unsigned team);
    void assign(const double* points,const double* centers,uint32_t* labels,unsigned count);
};
struct Reference {
    std::vector<double> centers,sums;
    std::vector<uint32_t> counts,labels;
    double objective=0;
    Reference(unsigned n,unsigned k,unsigned d,const std::vector<double>& initial)
      :centers(initial),sums(k*d),counts(k),labels(n) {}
    void step(const double* points,unsigned n,unsigned k,unsigned d);
};
bool close(double a,double b);
}
