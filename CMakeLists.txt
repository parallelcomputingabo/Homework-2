cmake_minimum_required(VERSION 3.10)
project(optimized_matmul)

set(CMAKE_CXX_STANDARD 17)

# Detect if we're on Apple Silicon
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(HOMEBREW_PREFIX "/opt/homebrew")
else()
    set(HOMEBREW_PREFIX "/usr/local")
endif()

# Set up OpenMP support for macOS
if(APPLE)
    message(STATUS "macOS detected, using Homebrew path: ${HOMEBREW_PREFIX}")
    
    # Include directories for OpenMP headers
    include_directories(${HOMEBREW_PREFIX}/opt/libomp/include)
    
    # Add compiler flags for OpenMP
    add_compile_options(-Xpreprocessor -fopenmp)
    
    # Add linker flags for OpenMP
    link_directories(${HOMEBREW_PREFIX}/opt/libomp/lib)
    link_libraries(omp)
else()
    # Standard OpenMP setup for other platforms
    find_package(OpenMP REQUIRED)
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_libraries(${OpenMP_CXX_LIBRARIES})
endif()

# Set executable output path to build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Copy data directory to build directory 
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Add optimization flags
add_compile_options(-O3 -march=native)

# Add the executable
add_executable(optimized_matmul src/main.cpp)