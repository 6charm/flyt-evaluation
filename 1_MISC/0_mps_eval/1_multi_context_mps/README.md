`nvcc -Xptxas -dlcm=cg multi_ctx.cu -o mps-bench -lineinfo -lcuda -lcudart`

```
 * 1. 
 * 
 * 2. MPS and CUDA compute modes: PROHIBITED, EXCLUSIVE_PROCESS, DEFAULT
 * 
 * 
 * =============================================================
 * 3. CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING: 
 * Should enable unique resource reservations for multiple cuda contexts in a process. 
 * -> a. MPS, single process, multi context
 * -> b. MPS, single process, multi-context, no MP_PARTITIONING.
 * =============================================================
 */

//  * 0. No MPS, all SMs, keep reducing and recording MPS limits. 
//  * 1. MPS with single process, single context, varying SM core allocs.
//  * 2. MPS with multi process, single context.
```


```
// Theoretical execution bounds:
// Calculate the FLOPS of the defined kernel
// then check out the flops/sec of the entire GPU.
// the active threads % will reduce this flops/sec number, 
// therefore execution time will increase for a given workload
// --
// Does this mean: Lower SM count will ALWAYS lead to increased execution time?
// no matter how light the workload?
// A: NO. This is a function of the kernel launch configuration
// that affects the degree of parallelism of the computation.
// If the launch config itself is less than the theoretical GPU
// thread capacity (#SMs * threads/SM), then reducing the %active threads wont matter
// upto a certain point.
// -HOWEVER, in the case where the GPU is fully saturated, i.e. 
// in total, there are more threads being launched than the GPU capacity,
// then reducing the % active threads will surely lead to slowdowns.
// ---
// Kernel Characteristics: Delays in kernel execution time arise due to 
// contention for shared resources (registers, memory) among various threads.
// Therefore, kernels with simple computations (using less registers) will
// support a higher degree of parallelism (more threads) before experiencing execution hits, 
// while kernels with heavy register usage will hit delays much sooner.
// it is useful to empirically calculate this bound for a particular GPU.
// =====
// How do these ideas relate to the current experiment?
// We need to show that a 50:50 SM split between two workloads
// will slow down the "heavier" workload and leave the "lighter"
// workload unchanged. 
// w_heavy: a workload that slows down when SM % is reduced from 75% to 50%
// w_light: a workload with latency unchanged when SM % is increased from 25% to 50%.
// ---
// w_kernel will have a certain LC at which it will begin experiencing 
// execution delays (this LC will reduce with SM core allocation, i.e. % threads)
// At the limiting LC thread count, a reduction in the SM Core allocation
// will trigger the first execution delays.
// Therefore the experiment will work as follows:
// w_heavy will be a workload with LC EQUAL to the <limiting LC value at any >= 75% thread allocation>.
// w_light will be a workload (same kernel) with LC LOWER than the 
// <limiting LC value at <=25% thread allocation>.
// When the resource allocation change occurs, it is clear that the w_heavy will experience delay
// while the w_light will perform as it was, or even better.
// ---
// Q. Consider the Matrix Multiplication Kernel. For a fixed launch
// configuration, why does execution time increase with a larger 
// matrix size? Assume no resource limits.
// A: After the input matrices have been copied onto the GPU DRAM, 
// computations (whatever the MM algorithm) is essentially every thread
// launched will bring a row and column into cache, then perform a 
// series of FMA until the result is calculated. 
// --> This FMA involves copying data from memory (cache or direct from DRAM) into registers, followed
// by the actual SP Unit calculation and writeback to GPU DRAM.
// --! LARGER MATRICES WILL REQUIRE EACH THREAD TO USE MORE REGISTERS
// --! PER OUTPUT RESULT CALCULATION, leading to contention and delays.
// We can think of two cases for the LC: 
// i) #warps 
```