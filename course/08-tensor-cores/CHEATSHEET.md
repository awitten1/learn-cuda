# Module 08 — Tensor cores

## Run
```
make 08-tensor-cores/wmma_hello
./08-tensor-cores/wmma_hello
```

## NCU metrics
```
ncu --metrics \
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__inst_executed_pipe_tensor.sum,\
  sm__cycles_active.avg \
  ./08-tensor-cores/wmma_hello
```
- `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`
  is the "are tensor cores busy?" number. Tiny kernels won't move it much;
  you'll really exercise this in Module 09 v7.

## SASS to look for
```
cuobjdump --dump-sass ./08-tensor-cores/wmma_hello
```
- `HMMA` — half-precision matrix multiply-accumulate.
- `IMMA` / `BMMA` — int / binary variants.
- On Blackwell sm_120 you may also see Q-prefixed (`QGMMA`/`QMMA`) for the
  newer asynchronous tensor-memory-accelerator instructions, depending on
  what nvcc emits for your WMMA call.

## Deliverable
- Confirm `wmma_hello` produces 16 in every output cell.
- Read the SASS and identify the `HMMA` (or successor) instruction.
- Optional: change `half` → `__nv_bfloat16` and re-measure.
