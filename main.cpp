#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <cstdint>
using namespace std;
void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    //TODO : Implement naive matrix multiplication
    fill(C, C + m * p, 0.0f);
    // Calculation of mult
    for(int i=0;i<m;i++)
        for(int j=0;j<p;j++)
            for(int k=0;k<n;k++)
                C[i*p+j]+=A[i*n+k]*B[k*p+j];
}

void blocked_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices
    // C = A * B
    fill(C, C + m * p, 0.0f);
    for (int ii = 0; ii < m; ii += block_size)
        for (int jj = 0; jj < p; jj += block_size)
            for (int kk = 0; kk < n; kk += block_size)
                // Process block: C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk+block_size] * B[kk:kk+block_size, jj:jj+block_size]
                    for (int i = ii; i < min(ii + block_size, m); i++)
                        for (int j = jj; j < min(jj + block_size, p); j++) {
                            float sum = 0.0;
                            for (int k = kk; k < min(kk + block_size, n); k++)
                                sum += A[i * n + k] * B[k * p + j];
                            C[i * p + j] += sum;
                        }
}

void parallel_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement parallel matrix multiplication using OpenMP
    // A is m x n, B is n x p, C is m x p


    #pragma omp parallel for
    for (int i = 0; i < m;i++) {
        for (int j = 0; j < p; j++) {
            float sum=0.0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * p + j];
            C[i * p + j]=sum;
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
   //TODO : Implement result validation
    ifstream fptr_result(result_file);
    ifstream fptr_refer(reference_file);

    try {
        if(fptr_result.fail())
            throw "result";
        else if(fptr_refer.fail())
            throw "reference";
    }
    catch(const char* name) {
        cout<<"Failed to read "<<name<<endl;
        exit(-1);
    }
    float temp1,temp2;
    while(!fptr_result.eof()) {
        fptr_result>>temp1;
        fptr_refer>>temp2;
        // I used deficit because of precision issue with floating numbers
        if(abs(temp1-temp2)>=0.5) {
            cout<<"Failed comparison: "<<temp1<<" "<<temp2<<endl;
            return false;
        }
    }
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
    std::string folder = "..//data//" + std::to_string(case_number) + "//";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";
    int m, n, p;  // A is m x n, B is n x p, C is m x p



    // TODO Read input0.raw (matrix A)
    ifstream fptr_input0(input0_file);
    fptr_input0>>m>>n;

    float* A = new float[m*n];
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            fptr_input0>>A[i*n+j];
    // TODO Read input1.raw (matrix B)
    ifstream fptr_input1(input1_file);
    fptr_input1>>n>>p;
    float* B = new float[n*p];
    for(int i=0;i<n;i++)
        for(int j=0;j<p;j++)
            fptr_input1>>B[i*p+j];
    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_blocked = new float[m * p];
    float *C_parallel = new float[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, A, B, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    // TODO Write naive result to file
    ofstream fptr_result(result_file);
    fptr_result << m<<" "<<p << endl;
    for(int i=0;i<m;fptr_result<<endl,i++)
        for(int j=0;j<p;j++)
            fptr_result<<C_naive[i*p+j]<<" ";
    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, A, B, m, n, p, 64);
    double blocked_time = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    fptr_result.close();
    fptr_result.clear();
    fptr_result.open(result_file);
    fptr_result << m<<" "<<p << endl;
    for(int i=0;i<m;fptr_result<<endl,i++)
        for(int j=0;j<p;j++)
            fptr_result<<C_blocked[i*p+j]<<" ";

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

    fptr_result.close();
    fptr_result.clear();
    fptr_result.open(result_file);
    fptr_result << m<<" "<<p << endl;
    for(int i=0;i<m;fptr_result<<endl,i++)
        for(int j=0;j<p;j++)
            fptr_result<<C_parallel[i*p+j]<<" ";
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