#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <sstream>

/**
 * @brief Performs naive matrix multiplication: C = A x B
 *
 * Multiplies two matrices A (m x n) and B (n x p) and stores the result in C (m x p).
 *
 * @param C Pointer to the result matrix.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A and rows in matrix B.
 * @param p Number of columns in matrix B.
 */
void naive_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p) {
    for (uint32_t i = 0; i < m; ++i) {
        // For each row of A
        for (uint32_t j = 0; j < p; ++j) {
            // For each column of B
            C[i * p + j] = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                // Sum dot product
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
            //Round the value to two decimals if the output has more than two decimals
            if (std::floor(C[i * p + j] * 100) != C[i * p + j] * 100) {
                C[i * p + j] = std::round(C[i * p + j] * 100.0f) / 100.0f;
            }
        }
    }
}

/**
 * @brief Performs blocked matrix multiplication: C = A x B
 *
 * Multiplies two matrices A (m x n) and B (n x p) using a blocked approach and stores the result in C (m x p).
 *
 * @param C Pointer to the result matrix.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A and rows in matrix B.
 * @param p Number of columns in matrix B.
 * @param block_size Size of the blocks used for multiplication.
 */

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    for (uint32_t ii = 0; ii < m; ii += block_size)
    for (uint32_t jj = 0; jj < p; jj += block_size)
        for (uint32_t kk = 0; kk < n; kk += block_size)
            for (uint32_t i = ii; i < std::min(ii + block_size, m); i++)
                for (uint32_t j = jj; j < std::min(jj + block_size, p); j++){
                    for (uint32_t k = kk; k < std::min(kk + block_size, n); k++)
                        C[i * p + j] += A[i * n + k] * B[k * p + j];

                    //Round the value to two decimals if the output has more than two decimals
                    if (std::floor(C[i * p + j] * 100) != C[i * p + j] * 100) {
                        C[i * p + j] = std::round(C[i * p + j] * 100.0f) / 100.0f;
                    }
                }
}

/**
 * @brief Performs naive matrix multiplication: C = A x B
 *
 * Multiplies two matrices A (m x n) and B (n x p) in a parallel way and stores the result in C (m x p).
 *
 * @param C Pointer to the result matrix.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A and rows in matrix B.
 * @param p Number of columns in matrix B.
 */
void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Set parallel region
    #pragma omp parallel for
    for (uint32_t i = 0; i < m; ++i) {
        // For each row of A
        for (uint32_t j = 0; j < p; ++j) {
            // For each column of B
            C[i * p + j] = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                // Sum dot product
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
            // Round the value to two decimals if the output has more than two decimals
            if (std::floor(C[i * p + j] * 100) != C[i * p + j] * 100) {
                C[i * p + j] = std::round(C[i * p + j] * 100.0f) / 100.0f;
            }
        }
    }
}

/**
 * @brief Writes the result matrix C to a file.
 *
 * Writes matrix dimensions and elements to the specified file.
 *
 * @param path File path for writing the result.
 * @param C Pointer to the result matrix.
 * @param m Number of rows in C.
 * @param p Number of columns in C.
 */
void write_result(const std::string& path, const float* C, int m, int p) {
    std::ofstream resultFile(path);
    // Handle error
    if (!resultFile) return; 

    // Write dimensions
    resultFile << m << " " << p << "\n"; 
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
             // Write element
            resultFile << C[i * p + j];
            if (j < p - 1) resultFile << " ";
        }
        resultFile << "\n";
    }
    resultFile.close();
}

/**
 * @brief Compares two matrix result files for exact match.
 *
 * Successful test will print "Files match exactly!".
 * Unsuccessful test will print the first mismatch found.
 *
 * @param result_file Path to the generated result matrix file.
 * @param reference_file Path to the expected output matrix file.
 */
bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream result(result_file);
    std::ifstream reference(reference_file);

    // Check if files opened successfully
    if (!result.is_open() || !reference.is_open()){
        std::cerr << "Validation: Error Opening File" << std::endl;
        return false;
    }

    int m_result = 0, p_result = 0;
    int m_output = 0, p_output = 0;
    std::string line;

    // Read dimensions from result.raw
    if (std::getline(result, line)) {
        std::stringstream ss(line);
        ss >> m_result >> p_result;
    }

    // Read dimensions from output.raw
    if (std::getline(reference, line)) {
        std::stringstream ss(line);
        ss >> m_output >> p_output;
    }

    // Check if dimensions match
    if (m_result != m_output || p_result != p_output) {
        std::cerr << "Validation: Matrix Dimensions Do Not Match!" << std::endl;
        return false;
    }

    // Compare matrices element by element
    float value_result = 0.0f, value_output = 0.0f;
    for (int i = 0; i < m_result; ++i) {
        for (int j = 0; j < p_result; ++j) {
            result >> value_result;
            reference >> value_output;
            if (value_result != value_output) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << value_result << " != " << value_output << std::endl;
                return false;
            }
        }
    }
    std::cout << "Files match exactly!" << std::endl;
    return true;
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

    int m = 0, n=0, p=0;
    std::string firstLine;

    // Read dimensions from input0.raw
    std::ifstream input0(input0_file); 
    std::ifstream input1(input1_file);

    // Check if files opened successfully
    if (!input0.is_open() || !input1.is_open()) {
        std::cerr << "Dimension Extraction: Error Opening File" << std::endl;
        return 1;
    }

    if (std::getline(input0, firstLine)) {
        std::stringstream ss(firstLine);
        std::string part1, part2;
        ss >> part1 >> part2;
        m = std::stoi(part1);
        n = std::stoi(part2);
    }

    // Read dimensions from input1.raw
    if (std::getline(input1, firstLine)) {
        std::stringstream ss(firstLine);
        std::string part1, part2;
        ss >> part1 >> part2;
        if (std::stoi(part1) != n) {
            std::cerr << "Dimension Extraction: Matrix Dimensions Do Not Match" << std::endl;
        }
        p = std::stoi(part2);
    }

    // Allocate memory for matrices A, B, and C using new or malloc
    float* matrixA = (float*)malloc(m * n * sizeof(float));
    float* matrixB = (float*)malloc(n * p * sizeof(float));

    std::string line;

    // Read matrix A (row-major order)
    for (int i = 0; i < m; i++) {
        std::getline(input0, line);

        std::stringstream ss(line);
        for (int j = 0; j < n; j++) {
            ss >> matrixA[i * n + j];
        }
    }

    // Read matrix B (row-major order)
    for (int i = 0; i < n; i++) {
        std::getline(input1, line);
        std::stringstream ss(line);
        for (int j = 0; j < p; j++) {
            ss >> matrixB[i * p + j];
        }
    }

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, matrixA, matrixB, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    write_result(result_file, C_naive, m, p);

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, matrixA, matrixB, m, n, p, 32);
    double blocked_time = omp_get_wtime() - start_time;

    // Write blocked result to file
    write_result(result_file, C_blocked, m, p);

    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel,matrixA, matrixB, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // Write parallel result to file
    write_result(result_file, C_parallel, m, p);

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
    free(matrixA);
    free(matrixB);
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}