#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <iomanip> // setprecision


// PERFORMANCE MEASUREMENT IS PROVIDED IN THE PERFORMANCE.md

// for testing
const uint32_t BLOCK_SIZE = 32;

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    //TODO : Implement naive matrix multiplication
        // A is m x n, B is n x p, C is m x p
        for (uint32_t i = 0; i < m; i++){
            for (uint32_t j = 0; j < p; j++){
                C[i * p + j] = 0.0f;
                for (uint32_t k = 0; k < n; k++){
                    C[i * p + j] += A[i * n + k] * B[k * p + j];
                }
                // rounding it
                C[i * p + j] = std::round(C[i * p + j] * 100.0f) / 100.0f;
            }
        }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    for (uint32_t i = 0; i < m * p; ++i){
        C[i] = 0.0f;
    }

    for (uint32_t ii = 0; ii< m; ii+= block_size){
        for (uint32_t jj = 0; jj < p; jj += block_size){
            for (uint32_t kk = 0; kk < n; kk += block_size){


                for (uint32_t i = ii; i < std::min(ii + block_size, m); ++i){
                    for (uint32_t j = jj; j < std::min(jj + block_size, p); ++j){
                        for (uint32_t k = kk; k < std::min(kk + block_size, n); ++k){
            
                            C[i * p + j] += A[i * n + k] * B[k * p + j];
                            
                        }
                    }
                }
            }
        }
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p
    // omp only uses integer so static_casted to it.
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(m * p); ++i){
        C[i] = 0.0f;
    }
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(m); i++){
        for (int j = 0; j < static_cast<int>(p); j++){
            for (int k = 0; k < static_cast<int>(n); k++){
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
   //TODO : Implement result validation
   // input is file path to result.raw and output.raw
    // read files line for line and compare
    // return true or false
    std::ifstream resultFile(result_file);
    std::ifstream outputFile(reference_file);

    int res_m, res_p, out_m, out_p;

    resultFile >> res_m >> res_p;
    outputFile >> out_m >> out_p;

    for (uint32_t i = 0; i < static_cast<uint32_t>(res_m); i++){
        for (uint32_t j = 0; j < static_cast<uint32_t>(res_p); j++){
            float result_val, expected_val;
            resultFile >> result_val;
            outputFile >> expected_val;

            //adding a small error tolerance for floating point because sometimes its not exact (421.51, 421.5)
            // const float error_tol = 0.000001f;

            if (result_val != expected_val){
                std::cerr << "error at " << i << ", " << j << ". Expected " << expected_val << ", got " << result_val << std::endl;
                return false;
            }

        }
    }
    std::cout << "Matrix is same, validation complete\n";
    return true;
}

void write_dot_after_value(std::ofstream& file, float value) {
    if (std::floor(value) == value) {
        file << std::fixed << std::setprecision(1) << value;
    } else {
        file << std::fixed << std::setprecision(2) << value;
    }
}

void write_matrix_to_file(const std::string &filename, float* C, uint32_t m, uint32_t p){
    std::ofstream result(filename);

    result << m << " " << p << "\n";
    for (uint32_t i = 0; i < m; ++i){
        for (uint32_t j = 0; j < p; ++j){
            write_dot_after_value(result, C[i * p + j]);
            if (j < p - 1){
                result << " ";
            }
        }
        result << "\n";
    }
    result.flush();
    result.close();
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
    std::string folder = "../data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    int m, n, p;

    // TODO Read input0.raw (matrix A)
    std::ifstream input0(input0_file);
    input0 >> m >> n;

    // TODO Read input1.raw (matrix B)
    std::ifstream input1(input1_file);
    input1 >> n >> p;

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // naive_matmul initialization

    // allocate memory for A and B
    float *A = new float[m*n];
    float *B = new float[n*p];

    // print dimension for A, B and C_naive
    std::cout << "A dimensions: " << m << " x " << n << std::endl;
    std::cout << "B dimensions: " << n << " x " << p << std::endl;
    std::cout << "C dimensions: " << m << " x " << p << std::endl;


    // Read elements to A and B
        // A matrix
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                input0 >> A[i* n + j];
            }
        }
        // B matrix
        for (int i = 0; i<n; i++){
            for (int j = 0; j < p; j++){
                input1 >> B[i*p + j];
            }
        }

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file
    write_matrix_to_file(result_file, C_naive, m, p);

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, BLOCK_SIZE);
    double blocked_time = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    write_matrix_to_file(result_file, C_blocked, m, p);

    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    write_matrix_to_file(result_file, C_parallel, m, p);

    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct) {
        std::cerr << "Parallel result validation failed for case " << case_number << std::endl;
    }

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds\n";
    std::cout << "Blocked time: " << blocked_time << " seconds\n";
    std::cout << "Parallel time: " << parallel_time << " seconds\n";
    std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    std::cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}