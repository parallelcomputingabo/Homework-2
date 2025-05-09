#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <string>
#include <iomanip>
using namespace std;

struct Matrix {
    int x, y;
    double* dataRowMajorOrder;
};

Matrix* getMatrix(string filePath) {
    // Read file
    ifstream file(filePath);

    if (!file) {
        Matrix* result = new Matrix;
        return result;
    }

    // Max size of a line (size can be anything if available in memory. 100000 in this case
    char line[100000];

    // Dimensions
    int x = 0;
    int y = 0;

    // iteration/list element
    int i = 0;

    // Matrix data, placed somewhere in memory
    double* collection = nullptr;

    while (file.getline(line, 100000)) {
        // Loop through the file, line by line.
        // A line is a character list consisting of 100000 slots

        // Split line by blank space into token
        char* token = strtok(line, " ");
        while (token != nullptr) {
            // Iterate over one token at a time

            // First line in each input file represents dimensions X Y 
            if (x == 0) {
                // If X has not been set, then pick first token from first line,
                // Assuming this is the first line and this is the first token
                x = atoi(token);
            }
            else if (y == 0) {
                // If Y has not been set, then pick second token from first line,
                // Assuming this is the first line and this is the second token iteration
                y = atoi(token);

                // Now, set the total slots (x * y) in memory 
                collection = new double[x * y];
            }
            else {
                // Insert token into collection
                // Assuming this is the second line and onwards
                collection[i++] = atof(token);
            }

            token = strtok(nullptr, " ");
        }
    }

    file.close();

    // Define the Matrix
    Matrix* result = new Matrix;

    // Matrix dimensions
    result->x = x;
    result->y = y;

    // Data collection of tokens from second line and onwards
    result->dataRowMajorOrder = collection;

    return result;
}


void naive_matmul(double* C, double* A, double* B, uint32_t m, uint32_t n, uint32_t p) {
    // A = First matrix, in the form of one-dimensional/row-major list
    // B = Second Matrix, in the form of one-dimensional/row-major list
    // C = A x B, in the form of one-dimensional/row-major list
    //
    // m = number of rows in matrix A
    // n = number of columns in matrix A
    // p = number of columns in matrix B

    // TODO: Implement naive matrix multiplication C = A x B
    // A is m x n, B is n x p, C is m x p

    // Just to keep track of what the assignment is saying...
    uint32_t N = p;

    /*
        Loop through A and B and perform multiplication for each row element in A with column element of B (m x p).
        Loop is multidimensional and covers elements as if they were part of matrices.

        The result C points to somewhere in the memory, where the calculations
    */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < N; j++) {

            C[i * N + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * N + j] += A[i * n + k] * B[k * N + j];
            }
        }
    }
}

void blocked_matmul(double* C, double* A, double* B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // TODO: Implement blocked matrix multiplication
    // A is m x n, B is n x p, C is m x p
    // Use block_size to divide matrices into submatrices

    // Decrease risks of cache misses and false sharing when operating sequentially.
    // This function utilize the principle of spatial locality
    // - Separate matrices into blocks containing a certain amount of elements
    // - These elements are closer to each other, and thus being fetched as cache line by the CPU
    // - This lowers the risks of wrong data elements being fetched from memory
    // - Because of the separating into blocks, the cache lines for each block are separated
    // This leads to the CPU cores/threads is more likely to work on their own blocks (the data stored in separate cache lines)
    // - Because of this, each core avoids false sharing because they fetch data blocks that are separate from each other
    // (they're less likely to work on the same data)
    for (int ii = 0; ii < m; ii += block_size) {
        for (int jj = 0; jj < p; jj += block_size) {
            for (int kk = 0; kk < n; kk += block_size) {

                // Loop through the rows of current block
                for (int i = ii; i < min(ii + block_size, m); ++i) {
                    
                    // Loop through the columns of current block
                    for (int j = jj; j < min(jj + block_size, p); ++j) {
                        // Perform multiplications (i x j)
                       
                        // Create C_aggregate, which is the sum
                        // of all products of multiplications of all elements (i x j)
                        // accordingly to basic math rules.

                        double C_aggregate = 0.0;

                        // if kk is not 0, then assign an initial value
                        if (kk != 0) {
                            C_aggregate = C[i * p + j];
                        }

                        // Loop through elements (located in this block span) of matrix A and B
                        // (e.g. SINGLE ELEMENT IN ROW i * ALL ELEMENTS IN COLUMN j)
                        for (int k = kk; k < min(kk + block_size, n); ++k) {
                            // Perform multiplication
                            C_aggregate += A[i * n + k] * B[k * p + j];
                        }
                       
                        // The actual product in the output matrice element is the sum of all products
                        // produced by multiplying i with all elements in column j:
                        C[i * p + j] = C_aggregate;
                    }

                    // Move on to next element in row i
                }
            }
        }
    }
}

void parallel_matmul(double *C, double *A, double *B, uint32_t m, uint32_t n, uint32_t p, int threads) {
    // A = First matrix, in the form of one-dimensional/row-major list
    // B = Second Matrix, in the form of one-dimensional/row-major list
    // C = A x B, in the form of one-dimensional/row-major list
    //
    // m = number of rows in matrix A
    // n = number of columns in matrix A
    // p = number of columns in matrix B

    // TODO: Implement naive matrix multiplication C = A x B
    // A is m x n, B is n x p, C is m x p

    // Just to keep track of what the assignment is saying...
    uint32_t N = p;

    /*
        Loop through A and B and perform multiplication for each row element in A with column element of B (m x p).
        Loop is multidimensional and covers elements as if they were part of matrices.

        The result C points to somewhere in the memory, where the calculations
    */

    // Set the number of threads to use for this parallel loop
    // If none is set, then use all threads
    if (threads > 0) {
        omp_set_num_threads(threads);
    }

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < N; j++) {

            C[i * N + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * N + j] += A[i * n + k] * B[k * N + j];
            }
        }
    }

}

bool compareFiles(string originPath, string targetPath) {
    ifstream originalFile(originPath);
    ifstream targetFile(targetPath);

    if (!originalFile || !targetFile) {
        return false;
    }

    string originalFileContents;
    string targetFileContents;

    while (getline(originalFile, originalFileContents) && getline(targetFile, targetFileContents)) {
        string originalFileLetters = "";
        string targetFileLetters = "";

        for (char character : targetFileContents) {
            if (character != ' ') {
                targetFileLetters += character;
            }
        }

        for (char character : originalFileContents) {
            if (character != ' ') {
                originalFileLetters += character;
            }
        }

        if (targetFileLetters != originalFileLetters) {
            targetFile.close();
            originalFile.close();
            return false;
        }
    }

    targetFile.close();
    originalFile.close();
    return true;
}


int main(int argc, char *argv[]) {
    string num = FOLDER_NUM;

    if (argc != 4) {
        std::cerr << "Usage: " << argc << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    int block_size = std::atoi(argv[2]);
    if (case_number < 0 || case_number > 11) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    // Construct file paths
    std::string folder = "C:\\Users\\Priva\\Documents\\DI-studies\\parallell-programming\\assignment2\\data\\" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string naive_result_file = folder + "naive_result.raw";
    std::string blocked_result_file = folder + "blocked_result.raw";
    std::string parallel_result_file = folder + "parallel_result.raw";
    std::string reference_file = folder + "output.raw";

    int m, n, p;
    // TODO Read input0.raw (matrix A)
    Matrix* AMatrix = getMatrix(input0_file);

    // TODO Read input1.raw (matrix B)
    Matrix* BMatrix = getMatrix(input1_file);

    m = AMatrix->x;
    n = AMatrix->y;
    p = BMatrix->y;

    int size = m * p;

    // Allocate memory for result matrices
    double *C_naive = new double[m * p];
    double *C_blocked = new double[m * p];
    double *C_parallel = new double[m * p];

    // Measure performance of naive_matmul
    double start_time = omp_get_wtime();
    naive_matmul(C_naive, AMatrix->dataRowMajorOrder, BMatrix->dataRowMajorOrder, m, n, p);
    double naive_time = omp_get_wtime() - start_time;

    


    // TODO Write naive result to file
    ofstream naiveResultFile(naive_result_file);
    naiveResultFile << m << " " << p << "\n";
    int i = 0;

    while (i < size) {
        bool isInteger = fabs(C_naive[i] - round(C_naive[i])) < 1e-6;

        if (i % p == p - 1 && i + 1 != size) {
            // If this is the last character in a line, then insert linebreaks
            // Some integers in the output file has a dot (e.g. 247.). This code makes sure to satisfy that in results.raw
            if (isInteger) {
                naiveResultFile << C_naive[i] << "." << "\n";
            }
            else {
                naiveResultFile << C_naive[i] << "\n";
            }
        }
        else {
            if (isInteger) {
                naiveResultFile << C_naive[i] << "." << " ";
            }
            else {
                naiveResultFile << C_naive[i] << " ";
            }
        }
        i++;
    }
    naiveResultFile.close();

    // Validate naive result
    bool naive_correct = compareFiles(reference_file, naive_result_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of blocked_matmul (use block_size = 32 as default)
    start_time = omp_get_wtime();
    blocked_matmul(C_blocked, AMatrix->dataRowMajorOrder, BMatrix->dataRowMajorOrder, m, n, p, block_size);
    double blocked_time = omp_get_wtime() - start_time;

    // TODO Write blocked result to file
    ofstream blockedResultFile(blocked_result_file);
    blockedResultFile << m << " " << p << "\n";
    i = 0;

    while (i < size) {
        bool isInteger = fabs(C_blocked[i] - round(C_blocked[i])) < 1e-6;

        if (i % p == p - 1 && i + 1 != size) {
            // If this is the last character in a line, then insert linebreaks
            // Some integers in the output file has a dot (e.g. 247.). This code makes sure to satisfy that in results.raw
            if (isInteger) {
                blockedResultFile << C_blocked[i] << "." << "\n";
            }
            else {
                blockedResultFile << C_blocked[i] << "\n";
            }
        }
        else {
            if (isInteger) {
                blockedResultFile << C_blocked[i] << "." << " ";
            }
            else {
                blockedResultFile << C_blocked[i] << " ";
            }
        }
        i++;
    }
    blockedResultFile.close();

    // Validate blocked result
    bool blocked_correct = compareFiles(reference_file, blocked_result_file);
    if (!blocked_correct) {
        std::cerr << "Blocked result validation failed for case " << case_number << std::endl;
    }

    // Measure performance of parallel_matmul
    start_time = omp_get_wtime();
    parallel_matmul(C_parallel, AMatrix->dataRowMajorOrder, BMatrix->dataRowMajorOrder, m, n, p, atoi(argv[3]));
    double parallel_time = omp_get_wtime() - start_time;

    // TODO Write parallel result to file
    ofstream parallelResultFile(parallel_result_file);
    parallelResultFile << m << " " << p << "\n";
    i = 0;

    while (i < size) {
        bool isInteger = fabs(C_parallel[i] - round(C_parallel[i])) < 1e-6;

        if (i % p == p - 1 && i + 1 != size) {
            // If this is the last character in a line, then insert linebreaks
            // Some integers in the output file has a dot (e.g. 247.). This code makes sure to satisfy that in results.raw
            if (isInteger) {
                parallelResultFile << C_parallel[i] << "." << "\n";
            }
            else {
                parallelResultFile << C_parallel[i] << "\n";
            }
        }
        else {
            if (isInteger) {
                parallelResultFile << C_parallel[i] << "." << " ";
            }
            else {
                parallelResultFile << C_parallel[i] << " ";
            }
        }
        i++;
    }
    parallelResultFile.close();

    // Validate parallel result
    bool parallel_correct = compareFiles(reference_file, parallel_result_file);
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
    delete[] AMatrix->dataRowMajorOrder;
    delete[] BMatrix->dataRowMajorOrder;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;
    delete[] AMatrix;
    delete[] BMatrix;

    return 0;
}