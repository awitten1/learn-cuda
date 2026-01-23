#!/bin/bash

set -eux

g++ -O3 -Wall -Wextra -Werror -march=native -std=c++17 sum.cpp -o sum_cpu
nvcc -O3 -Xcompiler=-march=native,-Wall,-Wextra,-Werror  -std=c++17 sum.cu -o sum_gpu
