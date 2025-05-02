cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Custom OpenMP setup for macOS
if(APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -D_Alignof=alignof")
	include_directories("/opt/homebrew/include")
	link_directories("/opt/homebrew/lib")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

add_executable(matmul main.cpp)

if(APPLE)
	target_link_libraries(matmul "-L/opt/homebrew/lib" "-lomp")
endif()