# Module 01 — Execution model

## Run
```
make 01-execution-model/smid_mapping 01-execution-model/launch_overhead
./01-execution-model/smid_mapping
./01-execution-model/launch_overhead
```

## Profile launch overhead with nsys
```
nsys profile -o launch_overhead --stats=true ./01-execution-model/launch_overhead
nsys stats launch_overhead.nsys-rep
```
Look at: `CUDA API Statistics` (cudaLaunchKernel time), `CUDA Kernel Statistics`
(per-kernel avg duration). Open `launch_overhead.nsys-rep` in Nsight Systems
GUI to see the timeline; zoom in until you see individual launches.

## Read SASS
```
cuobjdump --dump-sass ./01-execution-model/smid_mapping | less
```
Find the `S2R` instructions reading special registers (`SR_CTAID.X`,
`SR_TID.X`, `SR_VIRTID`). That's how `blockIdx`/`threadIdx`/`%smid` actually
get into registers.

## Deliverable
- Histogram of blocks-per-SM for `blocks = SMs`, `4 * SMs`, `1024`.
- Measured launch overhead with and without per-launch sync.
