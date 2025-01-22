# Flyt Primitives Experiments
Flyt implements a set of workflows towards its goal of exposing GPU endpoints 
for resource provisioning. The below experiments aim to measure their performance.

#### 0. Profiling GPU with compute-bound and memory-bound kernels
The idea is to induce warp-scheduler delays by varying launch configurations.

#### 1. Measuring overheads of Flyt-based API Virtualization
Record overheads of each component (libtirpc, vCUDA library, vServer, device) in the flyt virtualization path
for classes of CUDA API calls.

#### 2. Measuring overheads of VM-Migration
