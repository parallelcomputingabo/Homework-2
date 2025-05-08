#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

using matrix = std::vector<std::vector<double>>;

void reset_matrix(matrix &C, int m, int p) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            C[i][j] = 0.0;
}

bool are_equal(const matrix &X, const matrix &Y, int m, int p) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            if (X[i][j] != Y[i][j])
                return false;
    return true;
}

auto multiply_ijk(const matrix &A, const matrix &B, matrix &C, int m, int n, int p) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

auto multiply_ikj(const matrix &A, const matrix &B, matrix &C, int m, int n, int p) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < p; ++j)
                C[i][j] += A[i][k] * B[k][j];
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

void multiply_blocked(const matrix &A, const matrix &B, matrix &C, int m, int n, int p, int blockSize) {
    for (int ii = 0; ii < m; ii += blockSize)
        for (int jj = 0; jj < p; jj += blockSize)
            for (int kk = 0; kk < n; kk += blockSize)
                for (int i = ii; i < std::min(ii + blockSize, m); ++i)
                    for (int k = kk; k < std::min(kk + blockSize, n); ++k)
                        for (int j = jj; j < std::min(jj + blockSize, p); ++j)
                            C[i][j] += A[i][k] * B[k][j];
}

void multiply_blocked_omp(const matrix &A, const matrix &B, matrix &C, int m, int n, int p, int blockSize) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < m; ii += blockSize)
        for (int jj = 0; jj < p; jj += blockSize)
            for (int kk = 0; kk < n; kk += blockSize)
                for (int i = ii; i < std::min(ii + blockSize, m); ++i)
                    for (int k = kk; k < std::min(kk + blockSize, n); ++k)
                        for (int j = jj; j < std::min(jj + blockSize, p); ++j)
                            C[i][j] += A[i][k] * B[k][j];
}

void save_matrix_to_raw(const matrix &C, const std::string &folder, const std::string &filename, int m, int p) {
    if (!fs::exists(folder)) {
        fs::create_directory(folder);
    }

    std::string filepath = folder + "/" + filename;
    std::ofstream file(filepath);
    if (!file) {
        std::cerr << "Failed to Write: " << filepath << std::endl;
        return;
    }

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            file << C[i][j] << " ";
    file << "\n";
    file.close();
}

matrix read_matrix(const std::string &path, int &rows, int &cols) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open: " << path << std::endl;
        exit(1);
    }

    file >> rows >> cols;
    matrix M(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            file >> M[i][j];
    return M;
}

void print_speedup(double naive_time, double blocked_time, double parallel_time) {
    double blocked_speedup = naive_time / blocked_time;
    double parallel_speedup = naive_time / parallel_time;
    std::cout << "Blocked Speedup: " << blocked_speedup << std::endl;
    std::cout << "Parallel Speedup: " << parallel_speedup << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <folder_number>" << std::endl;
        return 1;
    }

    int dir_num = std::stoi(argv[1]);
    std::string folder = "data/" + std::to_string(dir_num);

    // Construct the full path for input files based on the folder number
    std::string input0_file = folder + "/input0.raw";
    std::string input1_file = folder + "/input1.raw";

    int m, n, n2, p;
    matrix A = read_matrix(input0_file, m, n);
    matrix B = read_matrix(input1_file, n2, p);

    if (n != n2) {
        std::cerr << "Dimension mismatch: A is " << m << "x" << n << ", B is " << n2 << "x" << p << std::endl;
        return 1;
    }

    // Output the matrix size
    std::cout << "Matrix Size: " << m << " x " << p << std::endl;

    matrix C1(m, std::vector<double>(p));
    matrix C2(m, std::vector<double>(p));
    const int blockSize = 32;

    // IJK
    reset_matrix(C1, m, p);
    auto ijk_time = multiply_ijk(A, B, C1, m, n, p);
    std::cout << "IJK: " << ijk_time << " ms" << std::endl;
    save_matrix_to_raw(C1, folder, "result_ijk.raw", m, p);

    // IKJ
    reset_matrix(C2, m, p);
    auto ikj_time = multiply_ikj(A, B, C2, m, n, p);
    std::cout << "IKJ: " << ikj_time << " ms" << std::endl;
    save_matrix_to_raw(C2, folder, "result_ikj.raw", m, p);

    if (!are_equal(C1, C2, m, p)) {
        std::cerr << "Error: IJK and IKJ results do not match!" << std::endl;
        return 1;
    }

    // Blocked
    reset_matrix(C1, m, p);
    auto start = std::chrono::high_resolution_clock::now();
    multiply_blocked(A, B, C1, m, n, p, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    auto blocked_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Blocked: " << blocked_time << " ms" << std::endl;
    save_matrix_to_raw(C1, folder, "result_blocked.raw", m, p);

    // Blocked + OpenMP (default 1 thread)
    reset_matrix(C2, m, p);
    start = std::chrono::high_resolution_clock::now();
    multiply_blocked_omp(A, B, C2, m, n, p, blockSize);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Blocked + OpenMP: " << parallel_time << " ms" << std::endl;
    save_matrix_to_raw(C2, folder, "result_blocked_omp.raw", m, p);

    if (!are_equal(C1, C2, m, p)) {
        std::cerr << "Error: Blocked and Blocked+OpenMP results do not match!" << std::endl;
        return 1;
    }

    // Speedup Calculation
    print_speedup(ijk_time, blocked_time, parallel_time);

    // Experiment with multiple thread counts
    for (int threads = 1; threads <= 8; ++threads) {
        omp_set_num_threads(threads);
        reset_matrix(C2, m, p);
        start = std::chrono::high_resolution_clock::now();
        multiply_blocked_omp(A, B, C2, m, n, p, blockSize);
        end = std::chrono::high_resolution_clock::now();
        double time_with_threads = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Blocked + OpenMP with " << threads << " threads: " << time_with_threads << " ms" << std::endl;
    }

    return 0;
}
