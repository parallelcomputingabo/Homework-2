# Parallel Programming

**Åbo Akademi University, Information Technology Department**

**Instructor: Alireza Olama**

## Homework Assignment 2: Optimizing Matrix Multiplication in C++



#### Performance Measurement

For each test case (0 through 9 in the `data` folder):

- Report the times in a table in your submission README.md, including:
    - Test case number.
    - Matrix dimensions (m × n × p).
    - Wall clock time for each implementation (in seconds).
    - Speedup of blocked and parallel implementations over the naive implementation.

Example table format:

| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time (s) | Parallel Time (s) | Blocked Speedup | Parallel Speedup |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 0         | 512 × 512 × 512        | 2.345          | 0.987            | 0.543             | 2.38×           | 4.32×            |





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