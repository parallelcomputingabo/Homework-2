#!/bin/bash

# Name of your executable
EXEC=./matmul

# Output file for timing summary
OUTPUT_FILE=results_summary.txt

# Clear previous output
echo "Test Case | Dimensions       | Naive (s) | Blocked (s) | Parallel (s) | Blocked Speedup | Parallel Speedup" > $OUTPUT_FILE
echo "----------------------------------------------------------------------------------------------------------" >> $OUTPUT_FILE

for i in {0..9}
do
    echo "Running test case $i..."
    OUTPUT=$($EXEC $i)

    # Extract timings and dimensions using grep and awk
    DIM=$(echo "$OUTPUT" | grep "Case" | awk '{print $3}')
    NAIVE=$(echo "$OUTPUT" | grep "Naive time" | awk '{print $3}')
    BLOCKED=$(echo "$OUTPUT" | grep "Blocked time" | awk '{print $3}')
    PARALLEL=$(echo "$OUTPUT" | grep "Parallel time" | awk '{print $3}')
    BS=$(echo "$OUTPUT" | grep "Blocked speedup" | awk '{print $3}')
    PS=$(echo "$OUTPUT" | grep "Parallel speedup" | awk '{print $3}')

    printf "%-9s | %-15s | %-9s | %-11s | %-13s | %-15s | %-16s\n" "$i" "$DIM" "$NAIVE" "$BLOCKED" "$PARALLEL" "$BS" "$PS" >> $OUTPUT_FILE
done

echo -e "\n✅ All tests completed. Results saved in $OUTPUT_FILE."
