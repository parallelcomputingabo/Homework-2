# Homework 2

By Thai Nguyen.

## How to run

1. Clone this branch to your own folder
2. In the project's root folder, create a folder called __/build__
3. Open your terminal and navigate to the __/build__
4. Prepare CMAKE

```
cmake ..
```

5. Build execution file

```
cmake --build .
```

6. Navigate to __/Debug__
7. Run the program

```
matmul [CASE] [BLOCK SIZE] [THREADS]
```

[CASE] = case number (folder number in __/data__)
[BLOCK SIZE] = number of bytes (e.g. 16, 32, 64) used in blockwise/cache-friendly multiplication (only relevant in blocked_matmul() function)
[THREADS] = number of threads to use in parallel multiplication. 0 = Max threads (only relevant in parallel_matmul() function)

e.g. Run program, using case 9, 32 bytes in blockwise multiplication and all available threads in parallel multiplication:

```
matmul 9 32 0
```


## Explanation and findings

Much of the code is the same as in homework 1, but this time
functions for parallel multiplication and cache friendly multiplication
have been added.

Below are the performance results for each data folder/test case. Based on
my findings, parallel multiplication becomes feasible very early (at 128x64x128), and
cache optimized multiplication remains slower than normal multiplication.

I added a case 10 and 11, containing much larger matrices. Luckily, this time
it becomes apparent that block/cache friendly multiplication in my code
actually works at some point (large amount of data needed to see effect!).

I've tested with various number of threads and blocksizes. The sweet spot for my
PC are 32 bytes in block size, and max amount of threads in order for the results to make sense. 

| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time (s) | Parallel Time (s) | Blocked Speedup | Parallel Speedup |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 0         | 64 x 64 x 64           | 0.0012         | 0.0016           | 0.0024            | 0.76×           | 0.50×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 1         | 128 x 64 x 128         | 0.0050         | 0.0062           | 0.0055            | 0.81×           | 1.01×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 2         | 100 x 128 x 56         | 0.0034         | 0.0042           | 0.0040            | 0.81×           | 0.87×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 3         | 128x64x128             | 0.0050         | 0.0066           | 0.0047            | 0.76×           | 1.06×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 4         | 32x128x32              | 0.0006         | 0.0007           | 0.0019            | 0.83×           | 0.33×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 5         | 200x100x256            | 0.0243         | 0.0327           | 0.0151            | 0.76×           | 1.64×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 6         | 256x256x256            | 0.0878         | 0.1070           | 0.0512            | 0.82×           | 1.71×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 7         | 256x300x256            | 0.1051         | 0.1251           | 0.0456            | 0.84×           | 2.30×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 8         | 64x128x64              | 0.0025         | 0.0032           | 0.0028            | 0.70×           | 0.86×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 9         | 256x256x257            | 0.0826         | 0.1041           | 0.0484            | 0.79×           | 1.70×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 10        | 512x512x512            | 0.9335         | 0.8905           | 0.2717            | 1.05×           | 3.43×            |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 11        | 1024x1024x1024         | 7.7868         | 5.3085           | 0.9058            | 1.46×           | 8.59×            |

Exceeding 32 bytes in block calculating won't make any difference. Number of threads running in parallel will 
affect the parallel multiplication performance (easily distinguishable).

---

Reasoning and explanations for the block and parallel multiplications
can be found in the code

**Computer specs**
OS: Windows 11
Processor: AMD Ryzen 7 7730U
RAM: 16GB @ 3200 MHz