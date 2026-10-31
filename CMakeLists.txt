cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(atlas_greengpu LANGUAGES CXX CUDA)
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_executable(atlas_smoke native/smoke/main.cu native/smoke/cpu.cpp)
set_target_properties(atlas_smoke PROPERTIES
  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_link_libraries(atlas_smoke PRIVATE OpenMP::OpenMP_CXX)
enable_testing()
add_test(NAME cpu_cuda_correctness COMMAND atlas_smoke)
set_tests_properties(cpu_cuda_correctness PROPERTIES TIMEOUT 60)

add_executable(atlas_measure native/measurement/main.cu)
set_target_properties(atlas_measure PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

find_package(Threads REQUIRED)
add_executable(atlas_static native/static/main.cu native/static/cpu.cpp)
set_target_properties(atlas_static PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_include_directories(atlas_static PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(atlas_static PRIVATE OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
add_test(NAME static_split_partial_block COMMAND atlas_static 4099 2049 64 3 4 1729)
set_tests_properties(static_split_partial_block PROPERTIES TIMEOUT 60)

add_executable(atlas_kmeans native/kmeans/main.cu native/kmeans/cpu.cpp)
set_target_properties(atlas_kmeans PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_include_directories(atlas_kmeans PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(atlas_kmeans PRIVATE OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(atlas_kmeans PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
add_test(NAME kmeans_migration COMMAND atlas_kmeans ${CMAKE_SOURCE_DIR}/tests/data/kmeans_ties.kmdat 3,1,0,3,0 5 4 fixed)
add_test(NAME kmeans_convergence COMMAND atlas_kmeans ${CMAKE_SOURCE_DIR}/tests/data/kmeans_ties.kmdat 1 20 4 converge)
set_tests_properties(kmeans_migration kmeans_convergence PROPERTIES TIMEOUT 60)

add_executable(atlas_hotspot native/hotspot/main.cu native/hotspot/cpu.cpp)
set_target_properties(atlas_hotspot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_include_directories(atlas_hotspot PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(atlas_hotspot PRIVATE OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(atlas_hotspot PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME hotspot_verify_migration COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/native_hotspot.py
  $<TARGET_FILE:atlas_hotspot> ${CMAKE_SOURCE_DIR}/tests/data/hotspot_edge.hsdat verify)
add_test(NAME hotspot_deferred_migration COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/native_hotspot.py
  $<TARGET_FILE:atlas_hotspot> ${CMAKE_SOURCE_DIR}/tests/data/hotspot_edge.hsdat timing)
set_tests_properties(hotspot_verify_migration hotspot_deferred_migration PROPERTIES TIMEOUT 90)
