#!/bin/bash
read -p "Enter argument for matmul: " ARG
cmake .
make
./matmul "$ARG"
