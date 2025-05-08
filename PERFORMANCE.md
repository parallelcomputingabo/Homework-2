
Test case with a block size=32 and thread amount = 2

| Test Case | Dimensions (m × n × p) | Naive Time (s) | Blocked Time size=32 (s) | Parallel Time threads=2 (s) | Blocked Speedup | Parallel Speedup |
|-----------|------------------------|----------------|--------------------------|-----------------------------|-----------------|------------------|
| 0         |64x64x64                |0.0014644       |0.0021207                 |0.0058374                    |0.690527x        |0.250865x         |
| 1         |128x64x128              |0.0060553       |0.0077893                 |0.0072098                    |0.777387x        |0.839871x         |
| 2         |100x128x56              |0.0032474       |0.0055423                 |0.0064217                    |0.58593x         |0.505692x         |
| 3         |128x64x128              |0.005331        |0.0106084                 |0.0066299                    |0.502526x        |0.804084x         |
| 4         |32x128x32               |0.0009075       |0.0013527                 |0.0055265                    |0.670881x        |0.164209x         |
| 5         |200x100x256             |0.0361264       |0.0427141                 |0.0146796                    |0.845772x        |2.46099x          |
| 6         |256x256x256             |0.104217        |0.127474                  |0.0432734                    |0.817551x        |2.40834x          |
| 7         |256x300x256             |0.136655        |0.148658                  |0.0588809                    |0.919258x        |2.32087x          |
| 8         |64x128x64               |0.0036189       |0.0053119                 |0.0051785                    |0.681282x        |0.698832x         |
| 9         |256x256x257             |0.0771411       |0.127632                  |0.0356897                    |0.604404x        |2.16144x          |


The parallel results varied a bit from run to run, because my computer is old. Sometimes rarely it was 3x speedup (with threads=2), and most of the times 2x. But the average results are recorded in the table.

With:
- threads=4, the test case 7 improved to the speedup 3.44913x, and other matrix dimension that were bigger improved to 3x speed, but testcase 9 stayed at 2x
- threads=8, the test case 7 improved to the speedup 3.10741x, but similar results
- threads=16, the test case 7 Parallel speedup: 3.00245x, another run 3.33392x
- threads=32,  decreased the overall results back to 2x

In conclusion, 4 is the optimal amount of threads.


Block Size 8

|Test Case|Time (s)|Speedup|
|---------|--------|-------|
|0        |0.0031317|0.599163x|
|1        |0.0119009|0.470536x|
|7        |0.175811|0.752659x|
|9        |0.146163| 0.57611x|

Block size 16

|Test Case|Time (s)|Speedup|
|---------|--------|-------|
|0        |0.0021967|0.613875x|
|1        |0.0104878|0.517296x|
|7        |0.165336|0.790999x|
|9        |0.131588|0.606085x|

Block size of 64 showed no improvement either, which means that the optimal block size for my specs are 32.

