cmake_minimum_required(VERSION 3.9)
project(matmul CXX)

find_package(OpenMP REQUIRED)
add_executable(matmul main.cpp)
target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX)
# or, if you prefer manual flags:
# target_compile_options(matmul PRIVATE -fopenmp)
# target_link_libraries(matmul PRIVATE -fopenmp)
