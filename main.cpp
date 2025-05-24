#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <windows.h>
#include <algorithm>
#include <iomanip>
#include <string.h>


//#define MAX_PATH 500

void write_result_to_file(const std::string& file_path, float* C, uint32_t m, uint32_t p);
void naive_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p);
void blocked_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size);
void parallel_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p);
std::string getExecutablePath();
std::string getParentPath();
bool fileExists(const std::string& path);
bool validate_result(const std::string& path);

double start_time, avg_time, elapsed_time;


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    std::string parentPath = getParentPath();
    std::cout << "Parent Path: " << parentPath << std::endl;

    std::string root_path = parentPath + "\\Homework-2\\data\\"; // Construct path relative to the parent directory
    std::cout << "Root Path: " << root_path << std::endl;


    // Construct file paths
    std::string folder = std::to_string(case_number) + "\\";
    std::string input0_file = root_path + folder + "input0.raw";
    std::string input1_file = root_path + folder + "input1.raw";
    std::string result_file = root_path + folder + "result.raw";
    std::string reference_file = root_path + folder + "output.raw";

    std::ifstream ifs;


    std::string input_dir_1 = root_path + folder + "input0.raw";
    std::string input_dir_2 = root_path + folder + "input1.raw";
    std::string result_dir = root_path + folder + "result.raw";

	std::cout << "input0 path dir: " << input_dir_1 << std::endl;

    std::ifstream input0(input_dir_1);
    std::ifstream input1(input_dir_2);
    if (!input0.is_open() || !input1.is_open()) {
        std::cerr << "Error opening input files." << std::endl;
        return -1;
    }

    int m, n;
    input0 >> m;
    input0 >> n;

    float* A = (float*)malloc(m * n * sizeof(float));
    if (!A) {
        std::cerr << "Memory allocation failed for matrix A." << std::endl;
        return -1;
    }

    // TODO Read input0.raw (matrix A)
    if (!input0.is_open())
    {
        std::cerr << "Error opening input0.raw" << std::endl;
        return -1;
    }

    for (int i = 0; i < m * n; ++i) input0 >> A[i];
    input0.close();

    int n_check, p;
    input1 >> n_check;
    input1 >> p;


    // TODO Read input1.raw (matrix B)
    if (!input1.is_open()) {
        std::cerr << "Error opening input1.raw" << std::endl;
        delete[] A;
        return -1;
    }
    float* B = new float[n * p];

    for (int i = 0; i < n * p; ++i) input1 >> B[i];
    input1.close();


    // Allocate memory for result matrices
    float* C_naive = new float[m * p];
    float* C_blocked = new float[m * p];
    float* C_parallel = new float[m * p];

    double naive_times[10], blocked_times[10], parallel_times[10], avg_naive_time, avg_blocked_time, avg_parallel_time,
        blocked_time, naive_time;

    for (int i = 0; i < 30; i++)
    {
        if (i < 10)
        {
            start_time = omp_get_wtime();
            naive_matmul(C_naive, A, B, m, n, p);
            naive_time = omp_get_wtime() - start_time;
            naive_times[i] = naive_time;
			write_result_to_file(result_file, C_naive, m, p);
            bool naive_correct = validate_result(root_path+folder);
            if (!naive_correct) {
                std::cerr << "iteration " << i << ": Naive result validation failed for case " << case_number << std::endl;
            }
            if (i == 9) std::cout << std::endl;

        }
        else if (i < 20)
        {
            const uint32_t block_sizes[] = { 16, 32, 64 };
            const int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
            for (int i = 0; i < num_block_sizes; ++i) {
                uint32_t block_size = block_sizes[i];
                start_time = omp_get_wtime();
                blocked_matmul(C_blocked, A, B, m, n, p, block_size);
                blocked_time = omp_get_wtime() - start_time;
                // TODO Write blocked result to file if needed

                // Validate blocked result
                write_result_to_file(result_file, C_blocked, m, p);

                bool blocked_correct = validate_result(root_path+folder);
                if (!blocked_correct) {
                    std::cerr << "Blocked result validation failed for block size " << block_size << " in case " << case_number << "\n" << std::endl;
                }

            }
            blocked_times[i - 10] = blocked_time;
            if (i == 19) std::cout << std::endl;
        }
        else
        {
            start_time = omp_get_wtime();
            parallel_matmul(C_parallel, A, B, m, n, p);
            double parallel_time = omp_get_wtime() - start_time;


            write_result_to_file(result_file, C_parallel, m, p);

            bool parallel_correct = validate_result(root_path+folder);
            if (!parallel_correct) {
                std::cerr << "Parallel result validation failed for case " << case_number << "\n" << std::endl;
            }
            parallel_times[i - 20] = parallel_time;
            if (i == 29) std::cout << std::endl;

        }
    }

    // Calculate average times
    avg_naive_time = 0.0;
    for (int i = 0; i < 10; i++) {
        avg_naive_time += naive_times[i];
    }
    avg_naive_time /= 10.0;

    avg_blocked_time = 0.0;
    for (int i = 0; i < 10; i++) {
        avg_blocked_time += blocked_times[i];
    }
    avg_blocked_time /= 10.0;

    avg_parallel_time = 0.0;
    for (int i = 0; i < 10; i++) {
        avg_parallel_time += parallel_times[i];
    }
    avg_parallel_time /= 10.0;
    // Print performance results

    std::cout << "\nCase " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Average Naive time: " << avg_naive_time << " seconds\n";
    std::cout << "Average Blocked time: " << avg_blocked_time << " seconds\n";
    std::cout << "Average Parallel time: " << avg_parallel_time << " seconds\n";
    std::cout << "Average Blocked speedup: " << (avg_naive_time / avg_blocked_time) << "x\n";
    std::cout << "Average Parallel speedup: " << (avg_naive_time / avg_parallel_time) << "x\n";



    // Clean up
    delete[] A;
    free(A);
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}

void naive_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {

    // for-loop to initialise result matrix C
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            C[i * p + j] = 0.0f;
        }
    }

    start_time = omp_get_wtime();

    // This nestled for-loop iterates first over rows of A, then columns of B
    // and finally columns of A and rows of B.
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            for (uint32_t k = 0; k < n; ++k) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }

    elapsed_time = omp_get_wtime() - start_time;
    std::cout << "Naive time: " << elapsed_time << " seconds" << std::endl;

}

void blocked_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // Initialize C to zero
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            C[i * p + j] = 0.0f;
        }
    }

    start_time = omp_get_wtime();

    // Blocked matrix multiplication
    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                // For each block
                uint32_t i_max = std::min<uint32_t>(ii + block_size, m);
                uint32_t j_max = std::min<uint32_t>(jj + block_size, p);
                uint32_t k_max = std::min<uint32_t>(kk + block_size, n);
                for (uint32_t i = ii; i < i_max; ++i) {
                    for (uint32_t k = kk; k < k_max; ++k) {
                        float a_val = A[i * n + k];
                        for (uint32_t j = jj; j < j_max; ++j) {
                            C[i * p + j] += a_val * B[k * p + j];
                        }
                    }
                }
            }
        }
    }
    elapsed_time = omp_get_wtime() - start_time;
    std::cout << "Blocked time: " << elapsed_time << " seconds, " << "block size: " << block_size << std::endl;

}

void parallel_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p) {
    // Initialize C to zero
#pragma omp parallel for
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            C[i * p + j] = 0.0f;
        }
    }

    // Measure wall clock time for the multiplication
    start_time = omp_get_wtime();

    // Parallel matrix multiplication
#pragma omp parallel for
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            for (uint32_t k = 0; k < n; ++k) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }

    elapsed_time = omp_get_wtime() - start_time;
    std::cout << "Parallel time: " << elapsed_time << " seconds" << std::endl;
}


bool validate_result(const std::string& path) {
    std::string result_file = path + "\\result.raw";
    std::string output_file = path + "\\output.raw";

    std::ifstream result(result_file);
    std::ifstream output(output_file);

    if (!result.is_open() || !output.is_open()) {
        std::cerr << "Error opening result or output file in path: " << path << std::endl;
        return false;
    }

    std::string result_line, output_line;
    int line_number = 1; // To track which line has a mismatch

    while (std::getline(result, result_line) && std::getline(output, output_line)) {
        std::istringstream result_stream(result_line);
        std::istringstream output_stream(output_line);

        float result_value, output_value;
        int value_index = 1; // To track which value in the line has a mismatch

        while (result_stream >> result_value && output_stream >> output_value) {
            // Compare the float values with a small tolerance to account for floating-point precision issues
            if (std::abs(result_value - output_value) > 1e-5) {
                std::cerr << "Mismatch found at line " << line_number << ", value " << value_index << ":\n"
                    << "Result: " << result_value << ", Output: " << output_value << std::endl;
                result.close();
                output.close();
                return false;
            }
            value_index++;
        }

        // Check if one line has more values than the other
        if ((result_stream >> result_value) || (output_stream >> output_value)) {
            std::cerr << "Mismatch in number of values at line " << line_number << "." << std::endl;
            result.close();
            output.close();
            return false;
        }

        line_number++;
    }

    return true;
}

std::string getExecutablePath() {
    char buffer[MAX_PATH];
    GetModuleFileName(NULL, buffer, MAX_PATH);
    std::string::size_type pos = std::string(buffer).find_last_of("\\/");
    return std::string(buffer).substr(0, pos);
}

bool fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.is_open();
}

std::string getParentPath() {
    char buffer[MAX_PATH];
    GetModuleFileName(NULL, buffer, MAX_PATH);
    std::string executablePath(buffer);
    std::string::size_type pos = executablePath.find_last_of("\\/");
    return executablePath.substr(0, pos).substr(0, executablePath.substr(0, pos).find_last_of("\\/"));
}

void write_result_to_file(const std::string& file_path, float* C, uint32_t m, uint32_t p) {
	std::ofstream ofs(file_path.c_str());
	if (!ofs.is_open()) {
		std::cerr << "Error opening file for writing: " << file_path << std::endl;
		return;
	}

    ofs << m << " " << p << std::endl;
    ofs << std::fixed << std::setprecision(2);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            ofs << C[i * p + j] << " ";
        }
        ofs << std::endl;
    }





	ofs.close();
}