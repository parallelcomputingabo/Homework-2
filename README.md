# Parallel Programming

**Åbo Akademi University, Information Technology Department**

**Author: Rifat Bin Monsur**

## Homework Assignment 2: Optimizing Matrix Multiplication in C++



### Performance Measurement Table

Tested out different block sizes and thread counts, 8 thread seemed to get the best result. Below table contains the measurement tested on 4 thread counts.

| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time (s) | Parallel Time (s) | Blocked Speedup      | Parallel Speedup     |
|-----------|------------------------|----------------|------------------|-------------------|----------------------|----------------------|
| 0         | 64x64x64               | 0              | 0.003            | 0.00039           | 0x                   | 0×                   |
| 1         | 128x64x128             | 0              | 0                | 0.0032            | Too fast to measure  | 0x                   |
| 2         | 100x128x56             | 0.003          | 0                | 0                 | Too fast to measure  | Too fast to measure  |
| 3         | 128x64x128             | 0.0031         | 0                | 0.0032            | Too fast to measure  | 0.999985x            |
| 4         | 32x128x32              | 0.0032         |                  | 0.0032            | Too fast to measure  | 1x                   |
| 5         | 200x100x256            | 0.0032         | 0.00339          | 0.0032            | 0.941181x            | 1x                   |
| 6         | 256x256x256            | 0.0192         | 0.0128           | 0.0032            | 1.75x                | 6.99x                |
| 7         | 256x300x256            | 0.022          | 0.0158           | 0.00359           | 1.39x                | 6.11x                |
| 8         | 64x128x64              | 0              | 0                | 0.00299           | Too fast to measure  | 0x                   |
| 9         | 256x256x257            | 0.017          | 0.0128           | 0.0034            | 1.32813x             | 4.99994x             |

---

