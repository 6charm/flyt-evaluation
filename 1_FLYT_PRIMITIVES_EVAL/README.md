Experimental Observations
---
1. At around 3200 warps (100 blocks, 1024 threads per block), kernel latencies begin increasing linearly.
--> Predicted limiter is much lower: 16 * 32 * 32 (# cores * #warps/core * #threads/warp) = 16000 threads (approx) -> 512 warps (16 blocks of 1024 threads)


Notes
-----
- nvidia-smi GPU utilization interpretation:
GPU busy is the percentage of time over the last second that any of the SMs was busy, and the memory utilization is actually the percentage of time the memory controller was busy during the last second. You can keep the utilization counts near 100% by simply running a kernel on a single SM and transferring 1 byte over PCI-E back and forth. Utilization is not a "how well you're using the resources" statistic but "if you're using the resources"

Profiling
Compile via `nvcc -Xptxas -dlcm=cg -o bench flyt_compute_bench.cu -lineinfo`
module load nsight-compute/...
ncu-cli --query-metrics
nsys profile --stats=true ./<file>
ncu --page details -f  -o cuda_bench.profout
ncu-ui