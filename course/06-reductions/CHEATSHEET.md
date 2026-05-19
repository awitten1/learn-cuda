# Module 06 — Reductions

## Run
```
make 06-reductions/reduce_v0_atomic
./06-reductions/reduce_v0_atomic
```

## NCU metrics for reductions
```
ncu --metrics \
  l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,\
  smsp__warp_issue_stalled_lg_throttle.pct,\
  smsp__warp_issue_stalled_long_scoreboard.pct,\
  sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
  gpu__time_duration.sum \
  ./06-reductions/reduce_v0_atomic
```
- v0 should be dominated by `lg_throttle` stalls and atomic sectors.
- v1 (shared) should reduce atomic sectors ~`grid`-fold.
- v2 (shuffle) should reduce shared-mem traffic AND atomics.

## SASS to look for
- `ATOM.E.ADD` — global atomic add. Many in v0, few in v1/v2.
- `RED.E.ADD` — reduction op (compiler may pick over ATOM when the return
  value is unused).
- `SHFL.DOWN` — warp shuffle. Watch them dominate the inner loop of v2.

## Stall reason taxonomy (memorize this)
- `long_scoreboard` = waiting on global memory.
- `short_scoreboard` = waiting on shared / MIO.
- `lg_throttle` = LSU pipe saturated (atomics will do it).
- `wait` = explicit `__syncthreads()` / fences.
- `not_selected` = there were other eligible warps, scheduler picked them.
- `barrier`, `membar`, `tex_throttle`, ...

## Deliverable
4-way table: GB/s of input read, dominant stall reason, atomic ops counted.
