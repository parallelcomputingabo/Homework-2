# Parallel Programming

**Åbo Akademi University, Information Technology Department**

**Instructor: Alireza Olama**

## Homework Assignment 2: Optimizing Matrix Multiplication in C++

**Due Date**: 08/05/2025

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

 Table:

| Test Case | Dimensions (m × n × p) | Naive Time (ms) | Blocked Time (ms) | Parallel Time (ms) |Blocked Speedup | Parallel Speedup |
|-----------|------------------------|-----------------|-------------------|--------------------|----------------|------------------|
| 0         | 64 x 64 x 64           | 0.7175          | 0.6494            | 0.8206             | 1.10487x       | 0.87436x         |
| 1         | 128 x 64 x 128         | 2.8666          | 2.6224            | 0.4038             | 1.09312x       | 7.09906x         |
| 2         | 100 x 128 x 56         | 1.9846          | 1.7484            | 0.3329             | 1.13509x       | 5.96155x         |
| 3         | 128 x 64 x 128         | 2.8636          | 2.6047            | 0.3826             | 1.0994x        | 7.48458x         |
| 4         | 32 x 128 x 32          | 0.3573          | 0.3191            | 0.0986             | 1.11971x       | 3.62373x         |
| 5         | 200 x 100 x 256        | 14.3102         | 12.5335           | 2.0792             | 1.14176x       | 6.88255x         |
| 6         | 256 x 256 x 256        | 46.7047         | 40.9841           | 5.5358             | 1.13958x       | 8.43685x         |
| 7         | 256 x 300 x 256        | 52.7779         | 47.3435           | 6.6947             | 1.11479x       | 7.88353x         |
| 8         | 64 x 128 x 64          | 1.412           | 1.2579            | 0.2205             | 1.12251x       | 6.40363x         |
| 9         | 256 x 256 x 257        | 45.6821         | 40.9893           | 5.4245             | 1.11449x       | 8.42144x         |



---
