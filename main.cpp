#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cstdint>

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < p; j++)
        {
            float sum = 0;
            for (uint32_t k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size)
{
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    for (uint32_t ii = 0; ii < m; ii += block_size)
        for (uint32_t jj = 0; jj < p; jj += block_size)
            for (uint32_t kk = 0; kk < n; kk += block_size)
                // Process block: C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk+block_size] * B[kk:kk+block_size, jj:jj+block_size]
                for (uint32_t i = ii; i < std::min(ii + block_size, m); i++)
                    for (uint32_t j = jj; j < std::min(jj + block_size, p); j++)
                    {
                        float sum = 0; // significantly faster than writing and reading C repeatedly
                        for (uint32_t k = kk; k < std::min(kk + block_size, n); k++)
                            sum += A[i * n + k] * B[k * p + j];
                        C[i * p + j] = sum;
                    }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
// A is m x n, B is n x p, C is m x p
#pragma omp parallel for
    for (uint32_t i = 0; i < m; i++)
        for (uint32_t j = 0; j < p; j++)
        {
            float sum = 0; // significantly faster than writing and reading C repeatedly
            for (uint32_t k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * p + j];
            C[i * p + j] = sum;
        }
}

bool validate_result(const std::string &result_file, const std::string &reference_file)
{
    FILE *a;
    FILE *b;

    a = fopen(result_file.c_str(), "r");
    b = fopen(reference_file.c_str(), "r");
    if (a == NULL)
    {
        fprintf(stderr, "Error opening %s\n", result_file.c_str());
        exit(1);
    }
    if (b == NULL)
    {
        fprintf(stderr, "Error opening %s\n", result_file.c_str());
        exit(1);
    }

    int len_a, len_b;
    fseek(a, SEEK_END, 0);
    fseek(b, SEEK_END, 0);
    len_a = ftell(a);
    len_b = ftell(b);
    fseek(a, SEEK_SET, 0);
    fseek(b, SEEK_SET, 0);

    if (len_a != len_b)
        return false;

    for (int i = 0; i < len_a; i++)
    {
        if (getc(a) != getc(b))
            return false;
    }

    return true;
}

/**
 * Formats a floating point number into `buf` to two-decimal precision and no trailing zeroes.
 */
void format_properly(float f, char *buf, size_t buf_size)
{
    int len = snprintf(buf, buf_size, "%.2f", f);
    char *ptr = buf + len - 1;
    while (*ptr == '0' && ptr >= buf)
    {
        *ptr-- = '\x00';
    }
}

void write_results(FILE *result, int m, int p, float *C)
{
    fprintf(result, "%d %d\n", m, p);
    char buf[16] = {0};
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            if (j > 0)
                fputc(' ', result);
            format_properly(C[i * p + j], buf, sizeof(buf));
            fprintf(result, "%s", buf);
        }
        if (i < m - 1)
            fputc('\n', result);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 2;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9)
    {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 2;
    }

    // Construct file paths
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    FILE *input0 = fopen(input0_file.c_str(), "r");
    if (input0 == NULL)
    {
        fprintf(stderr, "Error opening %s\n", input0_file.c_str());
        return 1;
    }
    FILE *input1 = fopen(input1_file.c_str(), "r");
    if (input1 == NULL)
    {
        fprintf(stderr, "Error opening %s\n", input1_file.c_str());
        return 1;
    }

    int m, n, p;
    fscanf(input0, "%d %d", &m, &n);
    fscanf(input1, "%d %d", &n, &p);

    float *A = new float[m * n];
    float *B = new float[n * p];

    float f = 0.0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fscanf(input0, "%f", &f);
            A[i * n + j] = f;
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < p; j++)
        {
            fscanf(input1, "%f", &f);
            B[i * p + j] = f;
        }
    }

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // NAIVE START
    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    FILE *result = fopen(result_file.c_str(), "w");
    if (result == NULL)
    {
        fprintf(stderr, "Error opening %s\n", result_file.c_str());
        return 1;
    }
    write_results(result, m, p, C_naive);
    fclose(result);

    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct)
    {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // BLOCKED START
    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 32);
    double blocked_time = omp_get_wtime() - start_time;

    result = fopen(result_file.c_str(), "w");
    if (result == NULL)
    {
        fprintf(stderr, "Error opening %s\n", result_file.c_str());
        return 1;
    }
    write_results(result, m, p, C_blocked);
    fclose(result);

    // Validate blocked result
    bool blocked_correct = validate_result(result_file, reference_file);
    if (!blocked_correct)
    {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // PARALLEL START
    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, A, B, m, n, p);
    double parallel_time = omp_get_wtime() - start_time;

    result = fopen(result_file.c_str(), "w");
    if (result == NULL)
    {
        fprintf(stderr, "Error opening %s\n", result_file.c_str());
        return 1;
    }
    write_results(result, m, p, C_parallel);
    fclose(result);

    // Validate parallel result
    bool parallel_correct = validate_result(result_file, reference_file);
    if (!parallel_correct)
    {
        std::cerr << "Parallel result validation failed for case " << case_number << std::endl;
    }

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds\n";
    std::cout << "Blocked time: " << blocked_time << " seconds\n";
    std::cout << "Parallel time: " << parallel_time << " seconds\n";
    std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    std::cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";

#ifdef PERFORMANCE_MD
    FILE *out;
    out = fopen("performance.md", "a");
    if (out != NULL)
    {
        if (ftell(out) == 0)
        {
            fprintf(out,
                    "| Test Case | Dimensions (m x n x p) | Naive Time (s) | Blocked Time (s) | Parallel Time (s) | Blocked Speedup | Parallel Speedup |\n"
                    "|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|\n");
        }
        fprintf(
            out,
            "| %-9d | %-3d x %-3d x %-10d | %-14f | %-16f | %-17f | %.3fx%-9s | %.3fx%-10s |\n",
            case_number,
            m, n, p,
            naive_time,
            blocked_time,
            parallel_time,
            (naive_time / blocked_time), "",
            (naive_time / parallel_time), "");
        fclose(out);
    }
    else
    {
        fprintf(stderr, "Error opening performance.md, skipped\n");
    }
#endif

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}
