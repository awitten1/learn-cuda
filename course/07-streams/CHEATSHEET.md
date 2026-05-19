# Module 07 — Streams & async copies

## Run
```
make 07-streams/pipeline
./07-streams/pipeline
```

## Nsight Systems is the right tool here
```
nsys profile -o pipe --stats=true -t cuda,nvtx,osrt ./07-streams/pipeline
nsys-ui pipe.nsys-rep        # GUI
```
What you're looking for in the timeline:
- Without streams: serial `[H2D][kernel][D2H]` bar.
- With streams:    `[H2D]` rows and `[kernel]` rows and `[D2H]` rows all
  active simultaneously (different streams = different rows).

Add NVTX ranges in code (`#include <nvtx3/nvToolsExt.h>`; link `-lnvToolsExt`)
to label phases on the timeline.

## Things that quietly defeat overlap
- Host buffer not pinned (`cudaMallocHost`) -- copies become synchronous.
- Using the default stream (stream 0) -- it synchronizes against everything.
- Too few chunks -- each one is too big to leave room for overlap.
- Calling `cudaMemcpy` (sync) instead of `cudaMemcpyAsync`.

## Deliverable
Two nsys timeline screenshots (serial vs streamed) and the speedup factor.
