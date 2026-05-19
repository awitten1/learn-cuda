# Module 09 — SGEMM capstone

The point of this module is iterative measurement. Don't write all the
versions and then profile -- profile after each one.

## Version ladder
| v  | change                                            | expected lesson                              |
|----|---------------------------------------------------|----------------------------------------------|
| v0 | naive 1-thread-per-output                         | memory bound, awful coalescing               |
| v1 | swap thread mapping so threadIdx.x → column       | reads of B coalesce; ~5-10x                  |
| v2 | shared-memory block tile (BM=BN=BK=32)            | reuse, ~5x more                              |
| v3 | + register tile (TM=TN=8 per thread)              | arithmetic intensity ↑, ~2-3x                |
| v4 | + vectorized 128-bit loads (float4 / LDG.128)     | fewer memory transactions                    |
| v5 | + double buffering (memcpy_async, cuda::pipeline) | hide gmem latency behind compute             |
| v6 | + careful occupancy + bank-conflict-free layout   | last-mile cleanup                            |
| v7 | tensor cores via wmma / mma.sync (fp16 / bf16)    | the real performance regime                  |
| v8 | cuBLAS reference (cublasGemmEx)                   | the ceiling                                  |

A realistic target on a 5060 Ti: v7 in the 60-80% of cuBLAS range. Don't
chase the last 5% -- diminishing returns and brittle code.

## Profile recipe (run after every version)
```
ncu --set full -o sgemm_v0 ./09-gemm/sgemm_v0_naive 4096
ncu -i sgemm_v0.ncu-rep --page details
```
Or compare versions side-by-side:
```
ncu --set full --import-source yes -o v0 ./09-gemm/sgemm_v0_naive 4096
ncu --set full --import-source yes -o v1 ./09-gemm/sgemm_v1_coalesced 4096
ncu --baseline v0.ncu-rep -i v1.ncu-rep
```

## Metrics that matter for GEMM
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` — are SMs busy?
- `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` —
  are tensor cores busy? (v7 only)
- `smsp__warp_issue_stalled_long_scoreboard.pct` — global mem latency stall.
  Should drop hard from v2 onward (shared mem reuse).
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — should drop ~`BK`-fold
  going from v0 → v2.
- `sm__sass_thread_inst_executed_op_fadd_pred_on.sum +
   sm__sass_thread_inst_executed_op_fmul_pred_on.sum +
   2*sm__sass_thread_inst_executed_op_ffma_pred_on.sum` ÷ time → measured
  FP32 GFLOPS. Compare to your card's theoretical peak.

## SASS sanity checks
- `LDG.E.128` should appear by v4. If you see only `LDG.E.32`, the compiler
  didn't vectorize -- check alignment.
- `LDS.128` should appear by v3-v4 (vectorized shared loads).
- `FFMA` should be the dominant arithmetic instruction in v3+.
- `HMMA` (or `QGMMA` on Blackwell) appears in v7.

## Deliverable
A chart: GFLOPS vs. version, with the cuBLAS line drawn across the top.
For each version, one sentence on what bottleneck you removed.
