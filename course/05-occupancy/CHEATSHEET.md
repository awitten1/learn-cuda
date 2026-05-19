# Module 05 — Occupancy

## Run
```
make 05-occupancy/occupancy_sweep
./05-occupancy/occupancy_sweep
```

The `--ptxas-options=-v` flag in the Makefile prints register/shared usage
at compile time. Read it. Example:
```
ptxas info    : Used 24 registers, ...
```

## NCU metrics
```
ncu --metrics \
  launch__registers_per_thread,\
  launch__block_size,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  smsp__warp_issue_stalled_long_scoreboard.pct,\
  smsp__warp_issue_stalled_short_scoreboard.pct \
  ./05-occupancy/occupancy_sweep
```
- `sm__warps_active.avg.pct_of_peak_sustained_active` is *measured* occupancy.
- The long/short scoreboard stalls indicate memory latency that more
  occupancy would hide (or not).

## Force register pressure
```
make NVCCFLAGS="-O3 -std=c++17 -lineinfo --ptxas-options=-v -Icommon -maxrregcount=32" \
     05-occupancy/occupancy_sweep
```
If you push too low, ptxas reports "stack frame" / "spill stores" -- spills
to local memory (which lives in DRAM). Almost always slower than the extra
occupancy you bought.

## Deliverable
Block-size sweep × { theoretical occupancy, measured occupancy, runtime }.
Then repeat with `-maxrregcount=32`. Was max occupancy ever fastest?
