# Flyt Primitives Experiments

0. Profiling GPU with compute-bound and memory-bound kernels
   
The idea is to induce warp-scheduler delays by varying launch configurations.

2. Measuring overheads of Flyt-based API Virtualization
   
Record overheads of each component (libtirpc, vCUDA library, vServer, device) in the flyt virtualization path
for classes of CUDA API calls.
