#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <algorithm>

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
	// Initialize C to zero
	for (uint32_t i = 0; i < m; i++) {
		for (uint32_t j = 0; j < p; j++) {
			C[i * p + j] = 0.0f;
		}
	}
	
	// Perform naive matrix multiplication using triple nested loop
	for (uint32_t i = 0; i < m; i++) {
		for (uint32_t j = 0; j < p; j++) {
			for (uint32_t k = 0; k < n; k++) {
				C[i * p + j] += A[i * n + k] * B[k * p + j];
			}
		}
	}
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
	// Initialize C to zero
	for (uint32_t i = 0; i < m; i++) {
		for (uint32_t j = 0; j < p; j++) {
			C[i * p + j] = 0.0f;
		}
	}
	
	// Perform blocked matrix multiplication
	for (uint32_t ii = 0; ii < m; ii += block_size) {
		for (uint32_t jj = 0; jj < p; jj += block_size) {
			for (uint32_t kk = 0; kk < n; kk += block_size) {
				// Process block
				// C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk+block_size] * B[kk:kk+block_size, jj:jj+block_size]
				for (uint32_t i = ii; i < std::min(ii + block_size, m); i++) {
					for (uint32_t j = jj; j < std::min(jj + block_size, p); j++) {
						for (uint32_t k = kk; k < std::min(kk + block_size, n); k++) {
							C[i * p + j] += A[i * n + k] * B[k * p + j];
						}
					}
				}
			}
		}
	}
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
	// Initialize C to zero
	#pragma omp parallel for
	for (uint32_t i = 0; i < m; i++) {
		for (uint32_t j = 0; j < p; j++) {
			C[i * p + j] = 0.0f;
		}
	}
	
	// Parallelize the outer loop with OpenMP
	#pragma omp parallel for
	for (uint32_t i = 0; i < m; i++) {
		for (uint32_t j = 0; j < p; j++) {
			float sum = 0.0f;
			for (uint32_t k = 0; k < n; k++) {
				sum += A[i * n + k] * B[k * p + j];
			}
			C[i * p + j] = sum;
		}
	}
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
	std::ifstream result(result_file, std::ios::binary);
	std::ifstream reference(reference_file, std::ios::binary);
	
	if (!result || !reference) {
		std::cerr << "Error opening result or reference file" << std::endl;
		return false;
	}
	
	// Determine file sizes
	result.seekg(0, std::ios::end);
	reference.seekg(0, std::ios::end);
	size_t result_size = result.tellg();
	size_t reference_size = reference.tellg();
	result.seekg(0, std::ios::beg);
	reference.seekg(0, std::ios::beg);
	
	// Check if file sizes match
	if (result_size != reference_size) {
		std::cerr << "Result and reference file sizes do not match" << std::endl;
		return false;
	}
	
	// Read and compare file contents
	const size_t buffer_size = 1024;
	float result_buffer[buffer_size];
	float reference_buffer[buffer_size];
	
	size_t total_elements = result_size / sizeof(float);
	size_t elements_compared = 0;
	
	const float epsilon = 1e-5f; // Tolerance for floating-point comparison
	
	while (elements_compared < total_elements) {
		size_t elements_to_read = std::min(buffer_size, total_elements - elements_compared);
		
		result.read(reinterpret_cast<char*>(result_buffer), elements_to_read * sizeof(float));
		reference.read(reinterpret_cast<char*>(reference_buffer), elements_to_read * sizeof(float));
		
		for (size_t i = 0; i < elements_to_read; i++) {
			if (std::abs(result_buffer[i] - reference_buffer[i]) > epsilon) {
				std::cerr << "Result differs from reference at element " << (elements_compared + i) << std::endl;
				std::cerr << "Result: " << result_buffer[i] << ", Reference: " << reference_buffer[i] << std::endl;
				return false;
			}
		}
		
		elements_compared += elements_to_read;
	}
	
	return true;
}

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <case_number> <naive|blocked|parallel>" << std::endl;
		return 1;
	}

	int case_number = std::atoi(argv[1]);
	std::string algorithm = argv[2];
	
	// Validate algorithm choice
	if (algorithm != "naive" && algorithm != "blocked" && algorithm != "parallel") {
		std::cerr << "Invalid algorithm: " << algorithm << std::endl;
		std::cerr << "Valid algorithms: naive, blocked, parallel" << std::endl;
		return 1;
	}
	
	if (case_number < 0 || case_number > 9) {
		std::cerr << "Case number must be between 0 and 9" << std::endl;
		return 1;
	}

	// Construct file paths
	std::string folder = "data/" + std::to_string(case_number) + "/";
	std::string input0_file = folder + "input0.raw";
	std::string input1_file = folder + "input1.raw";
	std::string result_file = "result.raw";  // Write to current directory
	std::string reference_file = folder + "output.raw";

	// Read input0.raw (matrix A)
	std::ifstream input0(input0_file, std::ios::binary);
	if (!input0) {
		std::cerr << "Error opening input0.raw" << std::endl;
		return 1;
	}
	
	// Read dimensions from input0.raw
	uint32_t m, n;
	input0.read(reinterpret_cast<char*>(&m), sizeof(uint32_t));
	input0.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
	
	// Allocate memory for A
	float *A = new float[m * n];
	input0.read(reinterpret_cast<char*>(A), m * n * sizeof(float));
	input0.close();
	
	// Read input1.raw (matrix B)
	std::ifstream input1(input1_file, std::ios::binary);
	if (!input1) {
		std::cerr << "Error opening input1.raw" << std::endl;
		delete[] A;
		return 1;
	}
	
	// Read dimensions from input1.raw
	uint32_t n_check, p;
	input1.read(reinterpret_cast<char*>(&n_check), sizeof(uint32_t));
	input1.read(reinterpret_cast<char*>(&p), sizeof(uint32_t));
	
	// Check if dimensions are compatible
	if (n != n_check) {
		std::cerr << "Incompatible matrix dimensions" << std::endl;
		delete[] A;
		return 1;
	}
	
	// Allocate memory for B
	float *B = new float[n * p];
	input1.read(reinterpret_cast<char*>(B), n * p * sizeof(float));
	input1.close();

	// Allocate memory for result matrix C
	float *C = new float[m * p];

	std::cout << "Matrix dimensions: A(" << m << "x" << n << "), B(" << n << "x" << p << ")" << std::endl;
	
	if (algorithm == "naive") {
		// Time naive matrix multiplication
		double t0 = omp_get_wtime();
		naive_matmul(C, A, B, m, n, p);
		double t1 = omp_get_wtime();
		std::cout << "Naive matrix multiplication time: " << (t1 - t0) * 1000 << " ms" << std::endl;
	} 
	else if (algorithm == "blocked") {
		// Time blocked matrix multiplication
		double t0 = omp_get_wtime();
		blocked_matmul(C, A, B, m, n, p, 32);  // Block size of 32
		double t1 = omp_get_wtime();
		std::cout << "Blocked matrix multiplication time: " << (t1 - t0) * 1000 << " ms" << std::endl;
	} 
	else if (algorithm == "parallel") {
		// Time parallel matrix multiplication
		double t0 = omp_get_wtime();
		parallel_matmul(C, A, B, m, n, p);
		double t1 = omp_get_wtime();
		std::cout << "Parallel matrix multiplication time: " << (t1 - t0) * 1000 << " ms" << std::endl;
	}

	// Write result to file
	std::ofstream output(result_file, std::ios::binary);
	if (!output) {
		std::cerr << "Error opening result file" << std::endl;
		delete[] A;
		delete[] B;
		delete[] C;
		return 1;
	}
	
	// Write dimensions to result file
	output.write(reinterpret_cast<char*>(&m), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&p), sizeof(uint32_t));
	
	// Write result matrix to file
	output.write(reinterpret_cast<char*>(C), m * p * sizeof(float));
	output.close();
	
	// Validate result
	if (validate_result(result_file, reference_file)) {
		std::cout << "Result validation successful!" << std::endl;
	} else {
		std::cerr << "Result validation failed!" << std::endl;
		delete[] A;
		delete[] B;
		delete[] C;
		return 1;
	}

	// Free memory
	delete[] A;
	delete[] B;
	delete[] C;
	
	return 0;
}