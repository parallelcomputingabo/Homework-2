#!/usr/bin/env bash
export OMP_NUM_THREADS=8

# Markdown table header
echo "| Case | Size (m×n×p) | Naive (s) | Blocked (s) | Blocked Speedup | Parallel (s) | Parallel Speedup |"
echo "|:----:|:------------:|:---------:|:-----------:|:---------------:|:------------:|:----------------:|"

for i in $(seq 0 9); do
  out=$(./matmul $i)

  # extract timings
  t_naive=$(echo "$out" | awk '/Naive matmul time/    {print $4}')
  t_block=$(echo "$out" | awk '/Blocked matmul time/  {print $5}')
  t_par=$(echo    "$out" | awk '/Parallel matmul time/ {print $4}')

  # read dimensions
  read m n < <(head -n1 ../data/$i/input0.raw)
  read _ p < <(head -n1 ../data/$i/input1.raw)

  # compute speedups
  sb=$(awk "BEGIN{printf \"%.2f×\", $t_naive/$t_block}")
  sp=$(awk "BEGIN{printf \"%.2f×\", $t_naive/$t_par}")

  # print markdown row
  printf "| %2d   | %4dx%4dx%4d | %7s  | %7s    | %8s        | %7s   | %8s      |\n" \
    $i $m $n $p \
    $t_naive \
    $t_block \
    "$sb" \
    $t_par \
    "$sp"
done
