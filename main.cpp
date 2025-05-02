#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <filesystem>



using namespace std;

bool validate_result(float* A, float* B, int rows, int cols) {
    // Compares two matrices A and B element-wise after rounding to 2 decimal places.
    // In the output.raw files, integers have a dot (e.g., 312.), but results.raw doesn't (e.g., 312)
    // So outputs are not EXACTLY same but this function ignores this difference by rounding both
    // values to two decimal places before comparing.
    // Returns true if all elements match; otherwise, prints failure and returns false.
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Round both elements to 2 decimal places to avoid floating-point precision errors
            float a = round(A[i * cols + j] * 100.0f) / 100.0f;
            float b = round(B[i * cols + j] * 100.0f) / 100.0f;

            if (a != b) {
                cout << "Validation Failed" << endl;
                return false;
            }
        }
    }
    cout << "Validation Passed" << endl;
    return true;
}

float* load_matrix(const string& path, int& rows, int& cols) {
    // Loads a matrix from a .raw file at the given path.
    // The first line of the file contains two integers: number of rows and columns.
    // The remaining values are the matrix elements in row-major order.
    // Returns a pointer to a dynamically allocated float array containing the matrix.
    // The dimensions are returned via reference parameters `rows` and `cols`.
    ifstream file(path);
    file >> rows >> cols;

    float* data = new float[rows * cols]; // Allocate memory for the matrix
    for (int i = 0; i < rows * cols; ++i) {
        file >> data[i];
    }

    file.close();
    return data;
}

void naive_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p) {
    // Performs naive matrix multiplication: C = A x B
    // A is an m x n matrix
    // B is an n x p matrix
    // C is the resulting m x p matrix, stored in row-major order
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = roundf(sum * 100.0f) / 100.0f;
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // Initialize output matrix to 0
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }

    // Blocked matrix multiplication
    for (uint32_t i0 = 0; i0 < m; i0 += block_size) {
        for (uint32_t j0 = 0; j0 < p; j0 += block_size) {
            for (uint32_t k0 = 0; k0 < n; k0 += block_size) {
                for (uint32_t i = i0; i < min(i0 + block_size, m); ++i) {
                    for (uint32_t j = j0; j < min(j0 + block_size, p); ++j) {
                        float sum = 0.0f;
                        for (uint32_t k = k0; k < min(k0 + block_size, n); ++k) {
                            sum += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] += sum;
                    }
                }
            }
        }
    }

    // round result like in naive_matmul
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = round(C[i] * 100.0f) / 100.0f;
    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Zero-out C in parallel
#pragma omp parallel for default(none) shared(C, m, p)
    for (uint32_t i = 0; i < m * p; ++i) {
        C[i] = 0.0f;
    }

    // Parallel matmul over rows
#pragma omp parallel for default(none) shared(A, B, C, m, n, p)
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = round(sum * 100.0f) / 100.0f;
        }
    }
}



int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }
    cout << "case_number: " << case_number << endl;
    // Construct file paths
    string folder = "../data/" + to_string(case_number) + "/";
    string input0_file = folder + "input0.raw";
    string input1_file = folder + "input1.raw";
    string result_file = folder + "result.raw";
    string reference_file = folder + "output.raw";

    cout << "Opening A from: " << input0_file << endl;
    cout << "Opening B from: " << input1_file << endl;
    cout << "Writing result to: " << result_file << endl;

    int m, n, p, dummy;  // A is m x n, B is n x p, C is m x p

    ifstream fileA(input0_file);
    ifstream fileB(input1_file);

    fileA >> m >> n;       // A is m x n
    fileB >> dummy >> p;   // B is n x p

    cout << "Dimensions of A: " << m << " x " << n << endl;
    cout << "Dimensions of B: " << dummy << " x " << p << endl;

    // TODO Read input0.raw (matrix A)
    float* A = new float[m * n];

    for (int i = 0; i < m * n; ++i) {
        fileA >> A[i];
    }
    fileA.close();


    // TODO Read input1.raw (matrix B)
    float* B = new float[n * p];

    for (int i = 0; i < n * p; ++i) {
        fileB >> B[i];
    }
    fileB.close();

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file
    ofstream fileC(result_file);

    // Write the matrix dimensions
    fileC << m << " " << p << endl;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            fileC << C_naive[i * p + j];
            if (j != p - 1) fileC << " ";
        }
        if (i != m - 1) fileC << endl;
    }

    fileC.close();


    // Validate naive result
    int ref_rows, ref_cols;
    float* ref_data = load_matrix(reference_file, ref_rows, ref_cols);

    int res_rows, res_cols;
    float* res_data = load_matrix(result_file, res_rows, res_cols);

    bool naive_correct = validate_result(res_data, ref_data, res_rows, res_cols);
    if (!naive_correct) {
        cerr << "Naive result validation failed for case " << case_number << endl;
    }

    delete[] ref_data;
    delete[] res_data;


    // Measure performance of blocked_matmul (use block_size = 32 as default)
    fill(C_blocked, C_blocked + (m * p), 0.0f);  // important: zero it out
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    // Write blocked result to file
    ofstream file_blocked(result_file);
    file_blocked << m << " " << p << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            file_blocked << C_blocked[i * p + j];
            if (j != p - 1) file_blocked << " ";
        }
        if (i != m - 1) file_blocked << endl;
    }
    file_blocked.close();

    // Validate blocked result
    int ref_rows2, ref_cols2;
    float* ref_blocked = load_matrix(reference_file, ref_rows2, ref_cols2);

    int res_rows2, res_cols2;
    float* res_blocked = load_matrix(result_file, res_rows2, res_cols2);

    bool blocked_correct = validate_result(res_blocked, ref_blocked, res_rows2, res_cols2);
    if (!blocked_correct) {
        cerr << "Blocked result validation failed for case " << case_number << endl;
    }

    delete[] ref_blocked;
    delete[] res_blocked;

    // Measure performance of parallel_matmul
    fill(C_parallel, C_parallel + (m * p), 0.0f);
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    ofstream file_parallel(result_file);
    file_parallel << m << " " << p << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            file_parallel << C_parallel[i * p + j];
            if (j != p - 1) file_parallel << " ";
        }
        if (i != m - 1) file_parallel << endl;
    }
    file_parallel.close();

    // Validate parallel result
    int ref_rows3, ref_cols3;
    float* ref_parallel = load_matrix(reference_file, ref_rows3, ref_cols3);

    int res_rows3, res_cols3;
    float* res_parallel = load_matrix(result_file, res_rows3, res_cols3);

    bool parallel_correct = validate_result(res_parallel, ref_parallel, res_rows3, res_cols3);
    if (!parallel_correct) {
        cerr << "Parallel result validation failed for case " << case_number << endl;
    }

    delete[] ref_parallel;
    delete[] res_parallel;

    // Print performance results
    cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    cout << "Naive time: " << naive_time << " seconds\n";
    cout << "Blocked time: " << blocked_time << " seconds\n";
    cout << "Parallel time: " << parallel_time << " seconds\n";
    cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}