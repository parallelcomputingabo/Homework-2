Platform: MacBook Air M1 chip.

Block size that gave the best speed up results: blocked_matmul with block size B=16 (I tried 32, and 64)

Fastest parallel execution was carried out with 8 threads (read the analysis below tables).

Note that for each test in the data folder, 3 result files are generated:  
- `result_naive.raw`  
- `result_blocked.raw`  
- `result_parallel.raw`


Building: Cmake was used for building using g++14,gcc 14.

Averaged results of 3 test rounds: 

| Case | Size (m×n×p)    | Naive (s) | Blocked (s) | Blocked Speedup | Parallel (s) | Parallel Speedup |
|:----:|:---------------:|:---------:|:-----------:|:---------------:|:------------:|:----------------:|
|  0   |    64×64×64     | 0.001784  |   0.001480  |   1.21×         |  0.000437    |   4.09×          |
|  1   |   128×64×128    | 0.003957  |   0.003650  |   1.08×         |  0.000848    |   4.73×          |
|  2   |   100×128×56    | 0.002369  |   0.002266  |   1.05×         |  0.000575    |   4.12×          |
|  3   |   128×64×128    | 0.003344  |   0.003310  |   1.01×         |  0.000824    |   4.06×          |
|  4   |    32×128×32    | 0.000434  |   0.000423  |   1.03×         |  0.000243    |   1.79×          |
|  5   |   200×100×256   | 0.017642  |   0.015881  |   1.11×         |  0.003823    |   4.63×          |
|  6   |   256×256×256   | 0.056921  |   0.053781  |   1.06×         |  0.011920    |   4.78×          |
|  7   |   256×300×256   | 0.066703  |   0.064336  |   1.04×         |  0.013929    |   4.79×          |
|  8   |    64×128×64    | 0.001738  |   0.001612  |   1.08×         |  0.000505    |   3.45×          |
|  9   |   256×256×257   | 0.056782  |   0.053043  |   1.07×         |  0.010905    |   5.21×          |


The individual 3 tests that were averaged are: 

Test 1: 
| Case | Size (m×n×p)    | Naive (s) | Blocked (s) | Blocked Speedup | Parallel (s) | Parallel Speedup |
|:----:|:---------------:|:---------:|:-----------:|:---------------:|:------------:|:----------------:|
|  0   |    64×64×64     | 0.001789  |   0.001478  |   1.21×         |  0.000408    |   4.38×          |
|  1   |   128×64×128    | 0.004026  |   0.003611  |   1.11×         |  0.000779    |   5.17×          |
|  2   |   100×128×56    | 0.002371  |   0.002211  |   1.07×         |  0.000582    |   4.07×          |
|  3   |   128×64×128    | 0.003341  |   0.003193  |   1.05×         |  0.000844    |   3.96×          |
|  4   |    32×128×32    | 0.000433  |   0.000437  |   0.99×         |  0.000243    |   1.78×          |
|  5   |   200×100×256   | 0.017643  |   0.015856  |   1.11×         |  0.004114    |   4.29×          |
|  6   |   256×256×256   | 0.056922  |   0.055528  |   1.03×         |  0.011895    |   4.79×          |
|  7   |   256×300×256   | 0.066691  |   0.064975  |   1.03×         |  0.013729    |   4.86×          |
|  8   |    64×128×64    | 0.001738  |   0.001630  |   1.07×         |  0.000513    |   3.39×          |
|  9   |   256×256×257   | 0.056785  |   0.055793  |   1.02×         |  0.010733    |   5.29×          |

Test 2: 
| Case | Size (m×n×p)    | Naive (s) | Blocked (s) | Blocked Speedup | Parallel (s) | Parallel Speedup |
|:----:|:---------------:|:---------:|:-----------:|:---------------:|:------------:|:----------------:|
|  0   |    64×64×64     | 0.001781  |   0.001472  |   1.21×         |  0.000461    |   3.86×          |
|  1   |   128×64×128    | 0.003886  |   0.003799  |   1.02×         |  0.000779    |   4.99×          |
|  2   |   100×128×56    | 0.002372  |   0.002400  |   0.99×         |  0.000569    |   4.17×          |
|  3   |   128×64×128    | 0.003346  |   0.003494  |   0.96×         |  0.000813    |   4.12×          |
|  4   |    32×128×32    | 0.000435  |   0.000395  |   1.10×         |  0.000237    |   1.84×          |
|  5   |   200×100×256   | 0.017641  |   0.015894  |   1.11×         |  0.003651    |   4.83×          |
|  6   |   256×256×256   | 0.056934  |   0.054274  |   1.05×         |  0.012246    |   4.65×          |
|  7   |   256×300×256   | 0.066701  |   0.064979  |   1.03×         |  0.013672    |   4.88×          |
|  8   |    64×128×64    | 0.001738  |   0.001615  |   1.08×         |  0.000473    |   3.67×          |
|  9   |   256×256×257   | 0.056767  |   0.051763  |   1.10×         |  0.010901    |   5.21×          |

Test 3:

| Case | Size (m×n×p)    | Naive (s) | Blocked (s) | Blocked Speedup | Parallel (s) | Parallel Speedup |
|:----:|:---------------:|:---------:|:-----------:|:---------------:|:------------:|:----------------:|
|  0   |    64×64×64     | 0.001782  |   0.001490  |   1.20×         |  0.000443    |   4.02×          |
|  1   |   128×64×128    | 0.003959  |   0.003541  |   1.12×         |  0.000985    |   4.02×          |
|  2   |   100×128×56    | 0.002365  |   0.002186  |   1.08×         |  0.000575    |   4.11×          |
|  3   |   128×64×128    | 0.003345  |   0.003244  |   1.03×         |  0.000815    |   4.10×          |
|  4   |    32×128×32    | 0.000433  |   0.000437  |   0.99×         |  0.000248    |   1.75×          |
|  5   |   200×100×256   | 0.017642  |   0.015892  |   1.11×         |  0.003705    |   4.76×          |
|  6   |   256×256×256   | 0.056906  |   0.051542  |   1.10×         |  0.011620    |   4.90×          |
|  7   |   256×300×256   | 0.066716  |   0.063053  |   1.06×         |  0.014385    |   4.64×          |
|  8   |    64×128×64    | 0.001737  |   0.001590  |   1.09×         |  0.000529    |   3.28×          |
|  9   |   256×256×257   | 0.056794  |   0.051574  |   1.10×         |  0.011080    |   5.13×          |


Results Analysis: 

Why I think blocked speed up was not much on Mac Air with M1 chip? I saw only a modest gain from blocking because the sizes of the test already fits in the M1’s caches (≈192 KB L1 + 12 MB L2), and its aggressive hardware prefetcher keeps the naive triple loop quite cache‐friendly. The extra loop nests and std::min calls introduce overhead that nearly cancels out the reduced cache misses.

To really benefit from tiling on the M1, I need much larger matrices (e.g. Case 9, 1024³) or a block size tuned to the L1 (32 or 64), or even a two-level blocking scheme or a hand-written microkernel that minimizes C loads/stores.


Why running on 8 cores yielded only ~4×–5× speedup on 8 threads: 

Heterogeneous cores
The M1 is 4 “Firestorm” performance cores + 4 “Icestorm” efficiency cores. When 8 threads are used, half of them land on the slower efficiency cores, so we don’t get an 8× boost—more like 5× on average.

Amdahl’s Law & overhead
Even though the inner (i,j) loops parallelize perfectly, there’s still:

A small serial fraction (startup, file I/O, validation).

OpenMP overhead (fork/join, scheduling).
That overhead eats into the ideal speedup.

Memory‐bandwidth limits
Matrix multiply is somewhat memory‐bound. On 8 cores, I start to saturate the shared L2 cache or the DRAM interface, so extra cores deliver diminishing returns.

Cache effects
While false sharing is not an issue here (each thread writes distinct C[i][j] regions), we do pay coherence traffic when threads read the same rows of A or columns of B. At high thread counts that can add latency.
