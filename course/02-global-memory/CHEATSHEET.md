# Module 02 — Global memory & coalescing

## Run
```
make 02-global-memory/copy_scalar
./02-global-memory/copy_scalar
```

## Nsight Compute metrics to look at
```
ncu --set full -o copy ./02-global-memory/copy_scalar
ncu -i copy.ncu-rep --page details
```
Or one-shot specific metrics:
```
ncu --metrics \
  dram__bytes_read.sum,dram__bytes_write.sum,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
  smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio,\
  gpu__time_duration.sum \
  ./02-global-memory/copy_scalar
```

Key ratios:
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` — should
  be 32 (perfect coalescing fills the 32-byte sector). Strided loads tank it.
- `l1tex__t_sector_hit_rate.pct` — L1 hit rate. For a streaming copy this
  should be near 0; if it's high you've sized things wrong.

## SASS to look for
```
cuobjdump --dump-sass ./02-global-memory/copy_scalar
```
- `LDG.E.SYS` — 32-bit global load.
- `LDG.E.128.SYS` — 128-bit (float4) load. You want these in copy_vec4.
- `STG.E` / `STG.E.128` — stores.

## DCGM while running
`dcgmi dmon -e 1009,1010,1011,1012` to see memcpy / mem-BW / PCIe utilization.

## Deliverable
Table:

| Variant        | ms | GB/s | % of D2D peak | sector ratio |
|----------------|----|------|---------------|--------------|
| scalar         |    |      |               |              |
| strided=2,4,8  |    |      |               |              |
| vec4           |    |      |               |              |
