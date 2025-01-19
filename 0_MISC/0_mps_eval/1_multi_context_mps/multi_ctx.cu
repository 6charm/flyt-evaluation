/**
 * FILE: multi_ctx.cu
 * ------------------
 * Test CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING:
 * "When non-uniform partitioning capability is enabled in an MPS client’s environment, 
 * client CUDA contexts can have different active thread percentages within 
 * the same client process via setting CUDA_MPS_ACTIVE_THREAD_PERCENTAGE before context creations. 
 * The device attribute cudaDevAttrMultiProcessorCount will reflect the active thread percentage and 
 * return the portion of available SMs that can be used by the client CUDA context current to the calling thread."
 * =====
 * a. For a process with multiple contexts, allows non-uniform resource limits per context. (without, each context gets same resource)
 * b. Without, each context must get the same resources. 
 * To demonstrate: Assign different workloads to different contexts. (Ensure the workload in context 1 
 * has max SM requirement if it has to meet an SLA. Then try a less intense workload in context 2, etc.)
 * ---> With CUDA_MPS_M_P, we can assign appropriate SMs to the larger workload etc, (SLA is met)
 * ---> without CUDA_MPS, we will observe that the kernel launch latency will fail the SLA.
 * =======
 * PROGRAM DESIGN
 * - Multiple pthreads, each with a unique context. Each thread submits a different kind of kernel and therefore
 * has unique SM requirements to meet an SLA. Without the multiproc_partition, the thread with 
 * highest SM req will fail the SLA (How to decide what this SLA must be?)
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>

static int active_device = 0;

void pr_gpu_cap(int &sm_cores, float &max_warps_per_sm_core, float &warp_size) {
    int device_count;
    cudaGetDeviceCount(&device_count);
    int device;
    for (device = 0; device < device_count; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        printf("GPU %d has compute capability %d.%d\n", device, prop.major, prop.minor);
        sm_cores = prop.multiProcessorCount;
        warp_size = (float)prop.warpSize;
        max_warps_per_sm_core = prop.maxThreadsPerMultiProcessor / (float)prop.warpSize;
        printf("GPU %d has %d SM Cores, max %.2f warps per core.\n",
               device, sm_cores, max_warps_per_sm_core);
    }
}

int change_sm_cores(uint32_t nm_sm_cores) {
    cudaDeviceSynchronize();

    CUcontext currentContext;
    CUresult res1;

    res1 = cuCtxGetCurrent(&currentContext);

    if (res1 != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res1, &errStr);
        return -1;
    }
    CUcontext newContext;
    CUresult res2;

    CUexecAffinityParam affinity_param;
    affinity_param.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinity_param.param.smCount.val = nm_sm_cores;

    // new cuda ctx with new sm_cores
    res2 = cuCtxCreate_v3(&newContext, &affinity_param, 1,  0, active_device);

    if (res2 != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res2, &errStr);
        return -1;
    }

    cuCtxPushCurrent(currentContext);
    cuCtxPopCurrent(&currentContext);
}

typedef void(*kernel_func_ptr)(int);

struct gpu_work {
    CUcontext ctx;
    pthread_t t_id;
    int sm_cnt;
    kernel_func_ptr k;
};

void launch_kernel(kernel_func_ptr _k, int N, int M, FILE *f_out) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int _iter = 1000;
    cudaEventRecord(start);
    _k<<<N, M>>>(_iter);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float t_ms = 0;
    cudaEventElapsedTime(&t_ms, start, stop);
    // printf("(%d, 1024): %.4f ms\n", N, t_ms);
    fprintf(f_out, "%d,%.4f\n", N,t_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void derive_SLA(kernel_func_ptr _k_heavy, kernel_func_ptr _k_light) {


}

// A 25:75 SM alloc vs a 50:50 SM alloc
// The light kernel needs to use [0%-25%] of SMs to meet the SLA.
// The heavy kernel needs to use [50%-75%] of its SMs to meet the SLA.
// ---> Latency SLA: The kernel exection time with the above SM reservations.
// When the sms are changed (25->50 and 75->50, the lighter kernel should be unaffected, 
// but the heavier kernel will slow down.)
// Find the min launch config such that 




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
#define THREAD_COUNT 2
int main(int argc,  char *argv[]) {
    derive_SLA(w_kernel);
    
}