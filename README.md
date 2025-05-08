# Matrix Multiplication Assignment (Parallel Programming)

This project implements and benchmarks three matrix multiplication methods:
- Naive Matrix Multiplication
- Blocked Matrix Multiplicatio
- Parallel Matrix Multiplication using OpenMP

#Objective
Evaluate and compare the performance of different matrix multiplication strategies on various test cases. The input matrices are read from files and results are validated against reference outputs.

---

# Project Structure

```
Assignment2/
├── build/              # Build directory (e.g., CMake or Visual Studio output)
├── data/               # Contains test case folders (0 to 9)
│   ├── 0/
│   │   ├── input0.raw  # Matrix A
│   │   ├── input1.raw  # Matrix B
│   │   ├── output.raw  # Reference result
│   │   └── result.raw  # Generated output
├── main.cpp            # Main implementation
├── CMakeLists.txt      # CMake build file (optional)
└── README.md           # This file
```

---

# How to Build

### 🧱 Using g++ (MinGW or Linux):
```bash
g++ -fopenmp -O2 main.cpp -o matmul
```

#On Windows (Visual Studio):
- Open project in Visual Studio.
- Enable OpenMP:  
  `Project Properties → C/C++ → Language → OpenMP Support → Yes (/openmp)`
- Build the solution.

---

## 🚀 How to Run

```bash
./matmul <case_number>
```

Example:
```bash
./matmul 3
```

Valid case numbers: `0` to `9`

---

#Output

For each case, the program prints:
- Matrix dimensions
- Execution time for each method
- Speedups over the naive method
- Validation status for result correctness

Example:
```
Case 3 (128x64x128):
Naive time: 0.003121 s
Blocked time: 0.006114 s
Parallel time: 0.002136 s
Blocked speedup: 0.51x
Parallel speedup: 1.46x
```

---

#Validation

The program compares computed `result.raw` with the reference `output.raw` and reports any mismatches. Modify `validate_result()` for custom error tolerances if needed.

---

#Notes

- Input and output matrices are in plain-text format.
- Block size for blocked multiplication is set to 32 by default.
- You can experiment with larger matrices and different block sizes for optimization.

---

# Author

This implementation is part of a Parallel Programming course assignment at Åbo Akademi University.

---

#License

This project is for educational purposes.