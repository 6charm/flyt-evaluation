# 1. Measuring overheads of Flyt-based API Virtualization
Record overheads of each component (libtirpc, vCUDA library, vServer, device) in the flyt virtualization path
for classes of CUDA API calls. The output must be a segmented bar graph with time spent during each phase
of computation.

`nvcc ovh.cu -O2 -arch sm_86 -o ovh -lcuda -cudart=shared -lrt -lpthread`

Flyt Version
---
Flyt vServer and vClient with timing API and shared-memory RPCs. Control plane any.

