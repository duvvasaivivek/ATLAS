#pragma once
#include <cstdint>
#include <vector>

namespace hs {
struct Model { double cap,rx,ry,rz,dt,nominal,ambient=80.,stability; };
Model model(unsigned rows,unsigned cols);
int64_t now();
void cpu_step(const double* old,double* next,const double* power,unsigned rows,unsigned cols,
              unsigned cut,unsigned workers,const Model& m,int64_t& begin,int64_t& end);
void reference_step(std::vector<long double>& old,std::vector<long double>& next,const double* power,
                    unsigned rows,unsigned cols,const Model& m);
}
