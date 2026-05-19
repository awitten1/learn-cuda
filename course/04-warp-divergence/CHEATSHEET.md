# Module 04 — Warp divergence

## Run
```
make 04-warp-divergence/divergence
./04-warp-divergence/divergence
```

## NCU metrics
```
ncu --metrics \
  smsp__thread_inst_executed_per_inst_executed.ratio,\
  smsp__warp_issue_stalled_branch_resolving.pct,\
  smsp__average_warps_active.ratio \
  ./04-warp-divergence/divergence
```
- `smsp__thread_inst_executed_per_inst_executed.ratio` is warp efficiency.
  32.0 = perfect (all 32 threads executing every instruction).
- `smsp__warp_issue_stalled_branch_resolving.pct` rises with divergence.

## SASS to look for
```
cuobjdump --dump-sass ./04-warp-divergence/divergence | less
```
Look at the `divergent_kernel` body:
- `@P0 ...` — predicated instructions (cheap, no branch).
- `BRA` — actual branches.
- `BSSY`/`BSYNC` (Volta+) — convergence barriers around divergent regions.

When `mod=1` or `mod>=32` the compiler typically predicates the whole switch
(no real branches). When `mod` is intermediate it can't, and you'll see
real `BRA` instructions and the serialization cost.

## Deliverable
Curve of warp efficiency and runtime vs `mod` ∈ {1,2,4,8,16,32,64}.
Explain the shape including the dip back down at mod=64.
