#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <omp.h>

// Naive matrix multiplication
void naive_matmul(float *C, const float *A, const float *B, int m, int n, int p)
{
    // Initialize C
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            C[i * p + j] = 0.0f;

    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            float a_val = A[i * n + k];
            for (int j = 0; j < p; ++j)
            {
                C[i * p + j] += a_val * B[k * p + j];
            }
        }
    }
}

// Blocked (tiled) matrix multiplication
void blocked_matmul(float *C, const float *A, const float *B,
                    int m, int n, int p, int block_size)
{
    // Initialize C
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            C[i * p + j] = 0.0f;

    for (int ii = 0; ii < m; ii += block_size)
    {
        for (int kk = 0; kk < n; kk += block_size)
        {
            for (int jj = 0; jj < p; jj += block_size)
            {
                int i_max = std::min(ii + block_size, m);
                int k_max = std::min(kk + block_size, n);
                int j_max = std::min(jj + block_size, p);
                for (int i = ii; i < i_max; ++i)
                {
                    for (int k = kk; k < k_max; ++k)
                    {
                        float a_val = A[i * n + k];
                        for (int j = jj; j < j_max; ++j)
                        {
                            C[i * p + j] += a_val * B[k * p + j];
                        }
                    }
                }
            }
        }
    }
}

// Parallel matrix multiplication with OpenMP
void parallel_matmul(float *C, const float *A, const float *B,
                     int m, int n, int p)
{
// Initialize C
#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            C[i * p + j] = 0.0f;
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
            {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// Validation (cent precision)
bool validate_result(const std::string &ref_path, const float *C, int m, int p)
{
    std::ifstream ref_file(ref_path);
    if (!ref_file)
    {
        std::cerr << "Error: Cannot open reference file " << ref_path << std::endl;
        return false;
    }
    int rm, rp;
    ref_file >> rm >> rp;
    if (rm != m || rp != p)
    {
        std::cerr << "Dimension mismatch: reference is " << rm << "x" << rp
                  << ", computed is " << m << "x" << p << std::endl;
        return false;
    }
    bool ok = true;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            float ref_val;
            ref_file >> ref_val;
            float comp_val = C[i * p + j];
            int ref_scaled = static_cast<int>(std::round(ref_val * 100.0f));
            int comp_scaled = static_cast<int>(std::round(comp_val * 100.0f));
            if (ref_scaled != comp_scaled)
            {
                std::cerr << "Mismatch at (" << i << "," << j << "): expected "
                          << ref_val << ", got " << comp_val << std::endl;
                ok = false;
            }
        }
    }
    return ok;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <case_number (0-9)>" << std::endl;
        return 1;
    }
    std::string case_id = argv[1];
    std::string folder = std::string("data/") + case_id + "/";
    std::string inputA_path = folder + "input0.raw";
    std::string inputB_path = folder + "input1.raw";
    std::string ref_path = folder + "output.raw";

    // Read A
    std::ifstream fileA(inputA_path);
    if (!fileA)
    {
        std::cerr << "Error opening " << inputA_path << std::endl;
        return 1;
    }
    int m, n;
    fileA >> m >> n;
    float *A = new float[m * n];
    for (int i = 0; i < m * n; ++i)
        fileA >> A[i];
    fileA.close();

    // Read B
    std::ifstream fileB(inputB_path);
    if (!fileB)
    {
        std::cerr << "Error opening " << inputB_path << std::endl;
        delete[] A;
        return 1;
    }
    int bn, p;
    fileB >> bn >> p;
    if (bn != n)
    {
        std::cerr << "Inner dimension mismatch" << std::endl;
        delete[] A;
        return 1;
    }
    float *B = new float[n * p];
    for (int i = 0; i < n * p; ++i)
        fileB >> B[i];
    fileB.close();

    // Allocate C
    float *C = new float[m * p];

    // Naive
    double t0 = omp_get_wtime();
    naive_matmul(C, A, B, m, n, p);
    double t1 = omp_get_wtime();
    bool ok_naive = validate_result(ref_path, C, m, p);
    std::cout << "Naive Time: " << (t1 - t0) << " s, "
              << (ok_naive ? "PASS" : "FAIL") << std::endl;

    // Blocked
    const int BLOCK_SIZE = 32; // adjust for best performance
    t0 = omp_get_wtime();
    blocked_matmul(C, A, B, m, n, p, BLOCK_SIZE);
    t1 = omp_get_wtime();
    bool ok_block = validate_result(ref_path, C, m, p);
    std::cout << "Blocked (" << BLOCK_SIZE << ") Time: " << (t1 - t0)
              << " s, " << (ok_block ? "PASS" : "FAIL") << std::endl;

    // Parallel
    t0 = omp_get_wtime();
    parallel_matmul(C, A, B, m, n, p);
    t1 = omp_get_wtime();
    bool ok_par = validate_result(ref_path, C, m, p);
    std::cout << "Parallel (threads=" << omp_get_max_threads() << ") Time: "
              << (t1 - t0) << " s, " << (ok_par ? "PASS" : "FAIL") << std::endl;

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
