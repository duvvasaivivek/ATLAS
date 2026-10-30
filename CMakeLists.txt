cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(atlas_phase1 LANGUAGES CXX CUDA)
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_executable(atlas_smoke native/smoke/main.cu native/smoke/cpu.cpp)
set_target_properties(atlas_smoke PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_link_libraries(atlas_smoke PRIVATE OpenMP::OpenMP_CXX)
enable_testing()
add_test(NAME cpu_cuda_correctness COMMAND atlas_smoke)
set_tests_properties(cpu_cuda_correctness PROPERTIES TIMEOUT 60)
