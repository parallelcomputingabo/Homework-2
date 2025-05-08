# Homework 2 – Optimizing Matrix Multiplication in C++

## Course
Parallel Programming – Spring 2025  
Åbo Akademi University

## Student
**Name:** Fahmida Khalid

---

## Overview

This project implements and benchmarks multiple matrix multiplication strategies to explore performance optimizations using cache blocking and OpenMP parallelism:

- **Naive (IJK order)** – Standard triple loop.
- **IKJ order** – More cache-friendly layout.
- **Blocked multiplication** – Improves cache reuse.
- **Blocked + OpenMP** – Parallel version using OpenMP with tunable thread counts.

Matrix data is read from `.raw` binary files, and results are validated and timed.

---

## Compilation

Ensure you have a compiler that supports OpenMP (e.g., `g++`):

```bash
g++ -fopenmp -O2 -o main main.cpp
```

---

## Usage

Run the program with a test index (0–9):

```bash
main <test_case>
```
main 0
main 1
.....
.....
main 9

Each index maps to a specific matrix dimension.

---

## Matrix Size Index Table

Matrix Size Index Table
Test Index	Matrix Size (m × n × p)
0	            64 x 64
1	            128 x 128
2	            100 x 56
3	            128 x 128
4           	32 x 32
5	            200 x 256
6	            256 x 256
7	            256 x 256
8	            64 x 64
9	            256 x 257
---

## Performance Results

Timings are in milliseconds. Speedup is computed relative to naive IJK implementation.

| Test Case | Dimensions (m × n × p) | Naive Time (ms) | Blocked Time (ms) | Parallel Time (ms) | Blocked Speedup | Parallel Speedup |
| --------- | ---------------------- | --------------- | ----------------- | ------------------ | --------------- | ---------------- |
| 0         | 64 x 64                | 2               | 3                 | 1                  | 0.67×           | 2.00×            |
| 1         | 128 x 128              | 10              | 11                | 3                  | 0.91×           | 3.33×            |
| 2         | 100 x 56               | 6               | 8                 | 3                  | 0.75×           | 2.00×            |
| 3         | 128 x 128              | 9               | 11                | 3                  | 0.82×           | 3.00×            |
| 4         | 32 x 32                | 2               | 1                 | 2                  | 2.00×           | 1.00×            |
| 5         | 200 x 256              | 47              | 55                | 12                 | 0.85×           | 3.92×            |
| 6         | 256 x 256              | 154             | 182               | 37                 | 0.85×           | 4.16×            |
| 7         | 256 x 256              | 185             | 215               | 46                 | 0.86×           | 4.02×            |
| 8         | 64 x 64                | 11              | 6                 | 2                  | 1.83×           | 5.50×            |
| 9         | 256 x 257              | 156             | 186               | 39                 | 0.84×           | 4.00×            |

---

## Observations

- The **blocked version** slightly underperforms in smaller matrices due to overhead but improves cache behavior in larger matrices.
- The **OpenMP parallel version** consistently outperforms all others from mid-sized matrices onward, reaching up to **4× speedup**.
- Best thread count is typically **4–6 threads**, with diminishing returns after that on this system.

---

## Parallel Scalability Example (Test 9 - 1200x1200)

| Threads | Time (ms) |
| ------- | --------- |
| 1       | 191       |
| 2       | 98        |
| 3       | 73        |
| 4       | 57        |
| 5       | 48        |
| 6       | 52        |
| 7       | 54        |
| 8       | 44        |


*Sweet spot: 4–5 threads before overhead limits gains.*

---

## Optimization Notes

- **Loop Reordering (IKJ)** improved locality.
- **Blocking (64x64)** enhanced cache reuse.
- **OpenMP** was applied to the outer loop of the blocked version for parallel execution.
- Performance was validated across 10 test sizes.

---


## Conclusion

This assignment demonstrates how algorithm design, memory layout, and parallelism all contribute to performance. OpenMP gave the highest speedup, but tuning block sizes and thread counts was essential for best results.