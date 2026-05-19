# learn-cuda course

A hardware-forward, profiler-heavy CUDA course targeting NVIDIA Blackwell
(RTX 5060 Ti, sm_120). Each module pairs a kernel with a measurement; don't
finish a module without numbers.

## Prerequisites
- CUDA Toolkit 12.8+ (for sm_120). Verify: `nvcc --version`.
- `nsys`, `ncu` (ship with the toolkit).
- `nvidia-smi`, `dcgmi`, optionally `nvtop`.
- A working C++17 toolchain.

## Build
```
cd course
make               # builds everything
make clean
make 02-global-memory/copy_scalar    # build one binary
```

The default arch is `sm_120`. Override with:
```
make ARCH=-arch=sm_89
```

`--ptxas-options=-v` is on by default; you'll see registers/shared usage at
compile time. Read it for every module.

## Module map
| #  | Topic                              | Hardware lesson                         |
|----|------------------------------------|-----------------------------------------|
| 00 | Survey                             | Spec sheet of your card, measured BW    |
| 01 | Execution model                    | Threads / warps / blocks / SMs / launch |
| 02 | Global memory                      | Coalescing, sectors, vectorized loads   |
| 03 | Shared memory                      | Tiling, bank conflicts                  |
| 04 | Warp divergence                    | Predication vs branching, efficiency    |
| 05 | Occupancy                          | Registers vs warps-in-flight tradeoff   |
| 06 | Reductions                         | Atomics, shared, warp shuffle, coop     |
| 07 | Streams                            | H<->D overlap, pinned memory, NVTX      |
| 08 | Tensor cores                       | WMMA, HMMA SASS, dtype throughput       |
| 09 | SGEMM capstone                     | Iterative path to ~80% of cuBLAS        |

## Suggested rhythm
1. Read the module starter `.cu` and its `CHEATSHEET.md`.
2. Predict the numbers before running.
3. Build + run, compare to prediction.
4. Run the NCU / nsys command from the cheatsheet.
5. Read the SASS for at least one kernel.
6. Fill in TODOs.
7. Write `notes/<module>.md` with what you measured and what surprised you.

## Monitoring in the background
Keep one of these in a second terminal while you work:
```
nvidia-smi dmon -s pucvmet -d 1
dcgmi dmon -e 1001,1002,1003,1004,1009,1010,1011,1012 -d 1000
nvtop
```
