# Parallel Programming

**Åbo Akademi University, Information Technology Department**

**Instructor: Alireza Olama**

## Homework Assignment 2: Optimizing Matrix Multiplication in C++



### Performance Measurement

For each test case (0 through 9 in the `data` folder):

### Experimenting with different block size (16,32,64)

| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time (s) | Block Size        | Blocked Speedup |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|
| 0         | 64 x 64 x 64           | 0.0014         | 0.0012           | 16                | 1.16×           | 
| 1         | 128 x 64 x 128         | 0.0035         | 0.0032           | 16                | 1.08×           | 
| 2         | 100 x 128 x 56         | 0.0024         | 0.0022           | 16                | 1.09×           | 
|-----------|------------------------|----------------|------------------|-------------------|-----------------|
| 0         | 64 x 64 x 64           | 0.0017         | 0.0014           | 32                | 1.15×           | 
| 1         | 128 x 64 x 128         | 0.0043         | 0.0037           | 32                | 1.17×           | 
| 2         | 100 x 128 x 56         | 0.0029         | 0.0025           | 32                | 1.19×           | 
| 3         | 100 x 64 x 128         | 0.0043         | 0.0035           | 32                | 1.21×           |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|
| 0         | 64 x 64 x 64           | 0.0017         | 0.0016           | 64                | 1.05×           | 
| 1         | 128 x 64 x 128         | 0.0033         | 0.0032           | 64                | 1.02×           | 
| 2         | 100 x 128 x 56         | 0.0027         | 0.0023           | 64                | 1.16×           | 
| 3         | 100 x 64 x 128         | 0.0037         | 0.0034           | 64                | 1.10×           |

Each test case were run 5 times and then calculated average is presented in the table. From the table I found the block multiplication algorithm works slightly well with 32 block size. Hence the optimal block size 32 is chosen for final comparison. 

Test code: [test_block.cpp](test_block.cpp)





### Experimenting with different threads (2,4,8)

| Test Case | Dimensions (m × n × p) | Naive Time (s) |Parallel Time (s) | Number of thread  |Parallel Speedup |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|
| 0         | 64 x 64 x 64           | 0.0008         | 0.0004           | 2                 | 1.73×           | 
| 1         | 128 x 64 x 128         | 0.0033         | 0.0020           | 2                 | 1.60×           | 
| 2         | 100 x 128 x 56         | 0.0024         | 0.0015           | 2                 | 1.53×           | 
|-----------|------------------------|----------------|------------------|-------------------|-----------------|
| 0         | 64 x 64 x 64           | 0.0010         | 0.0004           | 4                 | 2.43×           | 
| 1         | 128 x 64 x 128         | 0.0040         | 0.0015           | 4                 | 2.56×           | 
| 2         | 100 x 128 x 56         | 0.0026         | 0.0009           | 4                 | 2.72×           | 
| 3         | 100 x 64 x 128         | 0.0034         | 0.0012           | 4                 | 2.69×           |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|
| 0         | 64 x 64 x 64           | 0.0010         | 0.0016           | 8                 | 0.59×           | 
| 1         | 128 x 64 x 128         | 0.0042         | 0.0026           | 8                 | 1.59×           | 
| 2         | 100 x 128 x 56         | 0.0028         | 0.0049           | 8                 | 0.49×           | 
| 3         | 100 x 64 x 128         | 0.0061         | 0.0074           | 8                 | 0.81×           |

Similar to first one each test case is run 5 times and average is taken. Test code in [test_parrellel.cpp](test_parrellel.cpp) file.

With 8 threads the performance degrades, I am guessing cache coherence issues. So with 4 threads the function gives the best output. 




### Final Measurements

| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time (s) | Parallel Time (s) | Blocked Speedup | Parallel Speedup |
|-----------|------------------------|----------------|------------------|-------------------|-----------------|------------------|
| 0         | 64 x 64 x 64           | 0.0016         | 0.0015           | 0.0005            | 1.04×           | 2.97×            |
| 1         | 128 x 64 x 128         | 0.0038         | 0.0030           | 0.0013            | 1.27×           | 2.95×            |
| 2         | 100 x 128 x 56         | 0.0045         | 0.0021           | 0.0013            | 2.10×           | 3.50×            |
| 3         | 128 x 64 x 128         | 0.0069         | 0.0030           | 0.0017            | 2.26×           | 3.95×            |
| 4         | 32 x 128 x 32          | 0.0013         | 0.0012           | 0.0006            | 1.06×           | 2.23×            |
| 5         | 200 x 100 x 256        | 0.0017         | 0.0016           | 0.0004            | 1.06×           | 3.58×            |
| 6         | 256 x 256 x 256        | 0.0606         | 0.0547           | 0.0296            | 1.10×           | 2.04×            |
| 7         | 256 x 300 x 256        | 0.0627         | 0.0618           | 0.0336            | 1.01×           | 1.86×            |
| 8         | 64 x 128 x 64          | 0.0038         | 0.0021           | 0.0006            | 1.80×           | 5.93×            |
| 9         | 256 x 256 x 257        | 0.0508         | 0.0531           | 0.0161            | 0.95×           | 3.14×            |



  

    
      
      

**Parallel Matrix Multiplication with 4 threads performed better than Blocked Multiplication approach.**