# Module 00 — Survey

## Run
```
make 00-survey/device_query 00-survey/bandwidth_test
./00-survey/device_query
./00-survey/bandwidth_test
```

## Monitoring side-by-side (separate terminals)
```
watch -n 0.5 nvidia-smi
nvidia-smi dmon -s pucvmet -d 1          # power, util, clocks, mem, ecc, temp
dcgmi dmon -e 1001,1002,1003,1004,1009,1010,1011,1012 -d 1000
```

DCGM field IDs worth knowing:
- 1001 SM activity, 1002 SM occupancy, 1004 Tensor activity
- 1009 Mem copy util, 1010 Mem BW util
- 1011/1012 PCIe rx/tx bytes

## Deliverable
A `notes/00.md` with: SMs, regs/SM, shared/SM, L2, peak DRAM BW (theoretical
AND measured D2D), peak H2D pinned BW, observed power floor / idle clocks.
