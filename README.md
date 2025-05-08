# Parallel Programming

**Åbo Akademi University, Information Technology Department**

**Instructor: Alireza Olama**

## Homework Assignment 2: Optimizing Matrix Multiplication in C++

**Due Date**: 08/05/2025

**Points**: 100

---

### Assignment Overview

Welcome to the second homework assignment of the Parallel Programming course! In Assignment 1, you implemented a naive
matrix multiplication using a triple nested loop. In this assignment, you will optimize the performance of your naive
implementation using two techniques:

1. **Cache Optimization via Blocked Matrix Multiplication**: Improve data locality to reduce cache misses.
2. **Parallel Matrix Multiplication using `OpenMP`**: Parallelize the computation across multiple threads.

Your task is to implement both optimizations in the provided C++ `main.cpp` file, measure their performance, and compare the
wall clock time of the naive, cache-optimized, and parallel implementations for each test case. This assignment builds
on your Assignment 1 code, so ensure your naive implementation is correct before starting.

---

### Technical Requirements

#### 1. Cache Optimization (Blocked Matrix Multiplication)

**Why Cache Optimization?**

Modern CPUs rely on cache memory to reduce the latency of accessing data from main memory. Cache memory is faster but
smaller, organized in cache lines (typically 64 bytes). When a CPU accesses a memory location, it fetches an entire
cache line. Matrix multiplication can suffer from poor performance if memory accesses are not cache-friendly, leading to
frequent cache misses.

The naive matrix multiplication (with triple nested loops) accesses memory in a way that may not exploit spatial and
temporal locality:

- **Spatial Locality**: Accessing consecutive memory locations (e.g., elements in the same cache line).
- **Temporal Locality**: Reusing the same data multiple times while it’s still in the cache.

Blocked matrix multiplication divides the matrices into smaller submatrices (blocks) that fit into the cache. By
performing computations on these blocks, you ensure that data is reused while it resides in the cache, reducing cache
misses and improving performance.

**Blocked Matrix Multiplication Pseudocode**

Assume matrices \( A \) (m × n), \( B \) (n × p), and \( C \) (m × p) are stored in row-major order. The blocked matrix
multiplication processes submatrices of size \( block_size × block_size \):

```cpp
// C = A * B
for (ii = 0; ii < m; ii += block_size)
    for (jj = 0; jj < p; jj += block_size)
        for (kk = 0; kk < n; kk += block_size)
            // Process block: C[ii:ii+block_size, jj:jj+block_size] += A[ii:ii+block_size, kk:kk+block_size] * B[kk:kk+block_size, jj:jj+block_size]
            for (i = ii; i < min(ii + block_size, m); i++)
                for (j = jj; j < min(jj + block_size, p); j++)
                    for (k = kk; k < min(kk + block_size, n); k++)
                        C[i * p + j] += A[i * n + k] * B[k * p + j]
```

- **block_size**: Chosen to ensure the block fits in the cache (e.g., 32, 64, or 128, depending on the system).
- **Outer loops (ii, jj, kk)**: Iterate over blocks.
- **Inner loops (i, j, k)**: Compute within a block, reusing data in the cache.

**Task**: Implement the `blocked_matmul` function in the provided `main.cpp`. Experiment with different block sizes (e.g.,
16, 32, 64) and report the best performance.

---

#### 2. Parallel Matrix Multiplication with OpenMP

**Why OpenMP?**

`OpenMP` is a portable API for parallel programming in shared-memory systems. It allows you to parallelize loops with
minimal code changes, distributing iterations across multiple threads. In matrix multiplication, the outer loop(s) can
be parallelized, as each element of the output matrix \( C \) can be computed independently.

**Parallelizing with OpenMP**

Use OpenMP to parallelize the outer loop(s) of the naive matrix multiplication. For example, parallelize the loop over
rows of \( C \):

```cpp
#pragma omp parallel for
for (i = 0; i < m; i++)
    for (j = 0; j < p; j++)
        for (k = 0; k < n; k++)
            C[i * p + j] += A[i * n + k] * B[k * p + j];
```

- The `#pragma omp parallel for` directive tells `OpenMP` to distribute iterations of the loop across available threads.
- Ensure thread safety: Since each iteration writes to a distinct element of \( C \), this loop is safe to parallelize
  without locks.
- Use `omp_get_wtime()` to measure wall clock time for accurate performance comparisons.

**Task**: Implement the `parallel_matmul` function in the provided `main.cpp` using `OpenMP`. Test with different numbers of
threads (e.g., 2, 4, 8) by setting the environment variable `OMP_NUM_THREADS`.

---

#### 3. Performance Measurement

For each test case (0 through 9 in the `data` folder):

- Measure the **wall clock time** for:
    - Naive matrix multiplication (`naive_matmul`).
    - Cache-optimized matrix multiplication (`blocked_matmul`).
    - Parallel matrix multiplication (`parallel_matmul`).
- Use `omp_get_wtime()` for timing, as it provides high-resolution wall clock time.
- Report the times in a table in your submission README.md, including:
    - Test case number.
    - Matrix dimensions (m × n × p).
    - Wall clock time for each implementation (in seconds).
    - Speedup of blocked and parallel implementations over the naive implementation.

Example table format:

| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time (s) | Parallel Time (s) | Blocked Speedup | Parallel Speedup |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 0         | 512 × 512 × 512        | 2.345          | 0.987            | 0.543             | 2.38×           | 4.32×            |

---

| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time (s) | Parallel Time (s) | Blocked Speedup | Parallel Speedup |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 0         |      64 x 64 x 64      | 0.00074        | 0.00142          | 0.00265           | 0.52x           | 0.28x            |
| 1         |     128 x 64 x 12      | 0.00333        | 0.00490          | 0.00252           | 0.68x           | 1.32x            |
| 2         |     100 x 128 x 56     | 0.00191        | 0.00336          | 0.00173           | 0.57x           | 1.10x            |
| 3         |     128 x 64 x 12      | 0.00419        | 0.00613          | 0.00256           | 0.68x           | 1.64x            |
| 4         |      32 x 128 x 32     | 0.00035        | 0.00068          | 0.00304           | 0.52x           | 0.11x            |
| 5         |     200 x 100 x 25     | 0.01434        | 0.02394          | 0.01791           | 0.60x           | 0.80x            |
| 6         |     256 x 256 x 256    | 0.05017        | 0.08211          | 0.04476           | 0.61x           | 1.12x            |
| 7         |     256 x 300 x 256    | 0.05772        | 0.09756          | 0.05502           | 0.59x           | 1.05x            |
| 8         |      64 x 128 x 64     | 0.00185        | 0.00287          | 0.00130           | 0.64x           | 1.42x            |
| 9         |     265 x 256 x 257    | 0.04809        | 0.08143          | 0.04981           | 0.59x           | 0.97x            |


---

#### Matrix Storage and Memory Management

- Continue using row-major order for all matrices, as in Assignment 1.
- Use C-style arrays with manual memory management (`malloc` or `new`, `free` or `delete`).
- Do not use STL containers or smart pointers.

---

#### Input/Output and Validation

- Use the same input/output format as Assignment 1:
    - Input files: `data/<case>/input0.raw` (matrix \( A \)) and `input1.raw` (matrix \( B \)).
    - Output file: `data/<case>/result.raw` (matrix \( C \)).
    - Reference file: `data/<case>/output.raw` for validation.
- The executable accepts a case number (0–9) as a command-line argument.
- Validate correctness by comparing `result.raw` with `output.raw` for each implementation.

---

### Build Instructions

- Use the provided `CMakeLists.txt` to build the project.
- **Additional Requirements**:
    - Ensure OpenMP is enabled in your compiler (e.g., `-fopenmp` for GCC).
    - The provided CMake file includes OpenMP support.
- **Windows Users**:
    - Use CLion or Visual Studio with CMake.
    - Alternatively, use MinGW with `cmake -G "MinGW Makefiles"` and `make`.
- **Linux/Mac Users**:
    - Make sure gcc compiler is installed (`brew install gcc` on Mac).
    - Configure cmake to use the correct compiler:
      ```bash
      cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ .
      ```
    - Run `cmake .` to generate a Makefile, then `make`.
- **Testing OpenMP**:
    - Set the number of threads using the environment variable `OMP_NUM_THREADS` (e.g., `export OMP_NUM_THREADS=4` on
      Linux/Mac, or `set OMP_NUM_THREADS=4` on Windows).
    - Test with different thread counts to find the best performance.

---

### Submission Requirements

#### Fork and Clone the Repository

- Fork the Assignment 2 repository (provided separately).
- Clone your fork:
  ```bash
  git clone https://github.com/parallelcomputingabo/Homework-2.git
  cd Homework-2
  ```

#### Create a New Branch

```bash
git checkout -b student-name
```

#### Implement Your Solution

- Modify the provided `main.cpp` to implement `blocked_matmul` and `parallel_matmul`.
- Update `README.md` with your performance results table.

#### Commit and Push

```bash
git add .
git commit -m "student-name: Implemented optimized matrix multiplication"
git push origin student-name
```

#### Submit a Pull Request (PR)

- Create a pull request from your branch to the base repository’s `main` branch.
- Include a description of your optimizations and any challenges faced.

---

### Grading (100 Points Total)

| Subtask                                     | Points |
|---------------------------------------------|--------|
| Correct implementation of `blocked_matmul`  | 30     |
| Correct implementation of `parallel_matmul` | 30     |
| Accurate performance measurements           | 20     |
| Performance results table in README.md      | 10     |
| Code clarity, commenting, and organization  | 10     |
| **Total**                                   | 100    |

---

### Tips for Success

- **Cache Optimization**:
    - Experiment with different block sizes. Start with powers of 2 (e.g., 16, 32, 64).
    - Use a block size that balances cache usage without excessive overhead.
- **OpenMP**:
    - Test with different thread counts to find the optimal number for your system.
    - Be cautious of false sharing (when threads access nearby memory locations, causing cache coherence issues).
- **Performance Measurement**:
    - Run multiple iterations for each test case and report the average time to reduce variability.
    - Ensure no other heavy processes are running during measurements.
- **Debugging**:
    - Validate each implementation against `output.raw` to ensure correctness before optimizing.
    - Use small test cases to debug your blocked and parallel implementations.

Good luck, and enjoy optimizing your matrix multiplication!