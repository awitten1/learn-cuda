# Module 03 — Shared memory & bank conflicts

## Run
```
make 03-shared-memory/transpose_naive
./03-shared-memory/transpose_naive
```

## Key NCU metrics
```
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
  smsp__sass_l1tex_data_bank_conflicts.sum,\
  gpu__time_duration.sum \
  ./03-shared-memory/transpose_naive
```
- naive should have ~0 shared-mem conflicts (it doesn't use shared mem).
- tiled (no padding) should have many shared-mem store conflicts on TILE=32.
- tiled + `[TILE][TILE+1]` should have ~0 conflicts.

Also useful:
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` — should
  be 32 for the read in all three; the write is what differs.
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio` — naive
  near 4 (1 useful byte/sector), tiled near 32.

## SASS to look for
- `LDS` / `STS` — shared-memory load/store. Watch them appear in v1/v2.
- `BAR.SYNC` — the `__syncthreads()`.

## Deliverable
3 variants × { GB/s, shared bank conflicts, global st sector ratio }.
Confirm the +1 padding trick actually works on YOUR card.
