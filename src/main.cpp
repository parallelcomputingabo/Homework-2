#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm> // For std::min
#include <cmath>     // For std::abs, std::floor

#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
double omp_get_wtime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}
#endif

// Naive matrix multiplication C = A × B
void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Initialize C to zeros
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }
    
    // Triple loop implementation
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // Initialize C to zeros
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }
    
    // Blocked matrix multiplication
    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                // Process current block
                for (uint32_t i = ii; i < std::min(ii + block_size, m); ++i) {
                    for (uint32_t j = jj; j < std::min(jj + block_size, p); ++j) {
                        float sum = C[i * p + j]; // Load current value
                        for (uint32_t k = kk; k < std::min(kk + block_size, n); ++k) {
                            sum += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] = sum; // Store result back
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Initialize C to zeros
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }
    
    // Use parallel for on the outermost loop
    #pragma omp parallel for
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// Helper function to read matrix from file
bool read_matrix(const std::string &filename, float **matrix, uint32_t &rows, uint32_t &cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }
    
    // Read dimensions
    file >> rows >> cols;
    
    // Allocate memory
    *matrix = new float[rows * cols];
    
    // Read matrix data
    for (uint32_t i = 0; i < rows * cols; ++i) {
        file >> (*matrix)[i];
    }
    
    file.close();
    return true;
}

// Helper function to write matrix to file
bool write_matrix(const std::string &filename, float *matrix, uint32_t rows, uint32_t cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write dimensions
    file << rows << " " << cols << std::endl;
    
    // Write matrix data with precise formatting
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            float val = matrix[i * cols + j];
            
            // Format whole numbers as "XXX."
            if (val == std::floor(val)) {
                file << static_cast<int>(val) << ".";
            } else {
                // Format decimals with trailing zeros removed
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << val;
                std::string s = ss.str();
                s.erase(s.find_last_not_of('0') + 1, std::string::npos);
                if (s.back() == '.') s += "0";
                file << s;
            }
            
            if (j < cols - 1) file << " ";
        }
        file << std::endl;
    }
    
    file.close();
    return true;
}

// Function to compare floats with tolerance
bool compare_floats(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) <= tolerance;
}

// Function to compare lines containing space-separated floats
bool compare_float_lines(const std::string &line1, const std::string &line2, float tolerance = 0.001f) {
    std::istringstream iss1(line1), iss2(line2);
    float val1, val2;
    
    while (iss1 >> val1 && iss2 >> val2) {
        if (!compare_floats(val1, val2, tolerance)) {
            return false;
        }
    }
    
    // Check if both streams are exhausted
    return iss1.eof() && iss2.eof();
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream f1(result_file), f2(reference_file);
    if (!f1 || !f2) {
        std::cerr << "Error opening files for comparison" << std::endl;
        return false;
    }

    // Compare line by line
    std::string line1, line2;
    while (std::getline(f1, line1) && std::getline(f2, line2)) {
        if (line1 != line2 && !compare_float_lines(line1, line2)) {
            std::cerr << "Mismatch found:\n"
                      << "Expected: " << line2 << "\n"
                      << "Got:      " << line1 << std::endl;
            return false;
        }
    }

    return f1.eof() && f2.eof();
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }
    
    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }
    
    // Construct file paths
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";
    
    // Read input matrices
    float *A = nullptr;
    float *B = nullptr;
    uint32_t m, n, p, n_check;
    
    if (!read_matrix(input0_file, &A, m, n)) {
        std::cerr << "Failed to read matrix A" << std::endl;
        return 1;
    }
    
    if (!read_matrix(input1_file, &B, n_check, p)) {
        std::cerr << "Failed to read matrix B" << std::endl;
        delete[] A;
        return 1;
    }
    
    // Check matrix dimensions compatibility
    if (n != n_check) {
        std::cerr << "Matrix dimensions incompatible for multiplication" << std::endl;
        delete[] A;
        delete[] B;
        return 1;
    }
    
    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];
    
    // Display OpenMP information
    #ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cout << "Running with " << num_threads << " OpenMP threads" << std::endl;
    #else
    std::cout << "OpenMP not available. Running in single-threaded mode." << std::endl;
    #endif
    
    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;
    
    // Write naive result to file
    write_matrix(result_file, C_naive, m, p);
    
    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    std::cout << "Naive implementation: " << (naive_correct ? "PASSED" : "FAILED") << std::endl;
    
    // Test different block sizes for blocked implementation
    uint32_t block_sizes[] = {16, 32, 64, 128};
    double best_blocked_time = std::numeric_limits<double>::max();
    uint32_t best_block_size = 32; // Default
    
    for (uint32_t block_size : block_sizes) {
        start_time = omp_get_wtime();
        blocked_matmul(C_blocked, A, B, m, n, p, block_size);
        double blocked_time = omp_get_wtime() - start_time;
        
        write_matrix(result_file, C_blocked, m, p);
        bool blocked_correct = validate_result(result_file, reference_file);
        
        std::cout << "Blocked implementation (block_size=" << block_size << "): " 
                  << blocked_time << " seconds - " 
                  << (blocked_correct ? "PASSED" : "FAILED") << std::endl;
        
        if (blocked_time < best_blocked_time && blocked_correct) {
            best_blocked_time = blocked_time;
            best_block_size = block_size;
        }
    }
    
    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;
    
    // Write parallel result to file
    write_matrix(result_file, C_parallel, m, p);
    
    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file);
    std::cout << "Parallel implementation: " << (parallel_correct ? "PASSED" : "FAILED") << std::endl;
    
    // Implement a combined blocked+parallel implementation
    float *C_combined = new float[m * p];
    
    start_time = omp_get_wtime();
    // Initialize C to zeros
    for (uint32_t i = 0; i < m * p; ++i) {
        C_combined[i] = 0.0f;
    }
    
    // Combined blocked+parallel matrix multiplication
    #pragma omp parallel for collapse(2)
    for (uint32_t ii = 0; ii < m; ii += best_block_size) {
        for (uint32_t jj = 0; jj < p; jj += best_block_size) {
            for (uint32_t kk = 0; kk < n; kk += best_block_size) {
                // Process current block
                for (uint32_t i = ii; i < std::min(ii + best_block_size, m); ++i) {
                    for (uint32_t j = jj; j < std::min(jj + best_block_size, p); ++j) {
                        float sum = C_combined[i * p + j]; // Load current value
                        for (uint32_t k = kk; k < std::min(kk + best_block_size, n); ++k) {
                            sum += A[i * n + k] * B[k * p + j];
                        }
                        C_combined[i * p + j] = sum; // Store result back
                    }
                }
            }
        }
    }
    double combined_time = omp_get_wtime() - start_time;
    
    // Write combined result to file
    write_matrix(result_file, C_combined, m, p);
    
    // Validate combined result
    bool combined_correct = validate_result(result_file, reference_file);
    std::cout << "Combined Blocked+Parallel implementation: " << (combined_correct ? "PASSED" : "FAILED") << std::endl;
    
    // Print performance results
    std::cout << "\nCase " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds\n";
    std::cout << "Best Blocked time: " << best_blocked_time << " seconds (Block size: " << best_block_size << ")\n";
    std::cout << "Parallel time: " << parallel_time << " seconds\n";
    std::cout << "Combined time: " << combined_time << " seconds\n";
    std::cout << "Blocked speedup: " << (naive_time / best_blocked_time) << "x\n";
    std::cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";
    std::cout << "Combined speedup: " << (naive_time / combined_time) << "x\n";
    
    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;
    delete[] C_combined;
    
    return 0;
}