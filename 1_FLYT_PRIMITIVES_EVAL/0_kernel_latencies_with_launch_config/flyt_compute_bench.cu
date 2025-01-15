/**
 * FILE: flyt_compute_bench.cu
 * ---------------------------
 * A benchmark with adjustable parameters to 
 * deterministically saturating GPU compute.
 * 
 * The goal is to induce warp scheduler delays by modifying 
 * kernel launch configurations to increase occupancy.
 * ---> More threads -> more warps -> more delays
 * 
 * A set of dummy kernels are implemented to perform simple
 * compute tasks and memory tasks.
 * 
 * */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

// compute-only
__global__ void k_dummy_0(int _iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = tid;
    for (int i = 0; i < _iter; i++) {
        val = (val * 17 + i + _iter + tid) % 2048;
    }
}

// pure GPU DRAM only   
__global__ void k_dummy_1(int _iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int *global_data = new int[1024];  // per thread memory allocation.
    int val = tid;
    for (int i = 0; i < _iter; i++) {
        val = (global_data[tid % 1024] * 13 + i) % 1024;  // more memory access
    }
}

// pure shared-memory
__global__ void k_dummy_2(int _iter) {
    __shared__ int shared_data[1024];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = tid;
    for (int i = 0; i < _iter; i++) {
        shared_data[tid] = val * 13 + i;
        __syncthreads();
        val = shared_data[tid] % 1024;
    }
}

// Thread divergence compute only
__global__ void k_dummy_3(int _iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = tid;
    for (int i = 0; i < _iter; i++) {
        if (tid % 2 == 0) {
            val = (val * 13 + i + _iter) % 1024;
        } else {
            val = (val * 17 + i + _iter) % 1024;
        }
    }
}

// mixed
__global__ void k_dummy_4(int _iter) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int *global_data = new int[1024];  // Allocate global memory
    int val = tid;
    for (int i = 0; i < _iter; i++) {
        // Perform computation while also accessing memory
        val = (val * 13 + i) % 1024;
        global_data[tid % 1024] = val;  // Global memory write
    }
}

// heavier compute-only
__global__ void k_dummy_5(int _iter){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float value = tid * 1.0f;

    // Perform compute-intensive operations
    for (int i = 0; i < _iter; i++) {
        value = sinf(value) * cosf(value) + tanf(value) * value;
        value += sqrtf(fabsf(value));                           
        value = fmodf(value, 1000.0f);                          
    }
}

// Function to print GPU capabilities
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

typedef void(*kernel_func_ptr)(int);
int i = 0;
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

#define THREADS_PER_BLOCK 1024
int main(int argc, char *argv[]) {
    int sm_cores;
    float max_warps_per_sm_core, warp_size;

    pr_gpu_cap(sm_cores, max_warps_per_sm_core, warp_size);

    float warps_at_max_occupancy = sm_cores * max_warps_per_sm_core; // 16 * 32
    // choose kernel
    int _kern = atoi(argv[1]);
    kernel_func_ptr kernel = NULL;
    char f_name[128];
    sprintf(f_name,"cuda_bench_blocks_vs_latency_%s.csv", argv[1]);
    printf("%s\n", f_name);

    FILE *f_out = fopen(f_name, "w");
    switch(_kern) {
        case 0:
            kernel = k_dummy_0; // pure compute
            break;
        case 1:
            kernel = k_dummy_1; // pute DRAM 
            break;
        case 2:
            kernel = k_dummy_2; // pure shared memory
            break;
        case 3:
            kernel = k_dummy_3; // divergent, compute only
            break;
        case 4:
            kernel = k_dummy_4; // mixed compute and dram.
            break;
        case 5:
            kernel = k_dummy_5;
            break;
        default:
            printf("Kernel not implemented: %d\n", _kern);
            return EXIT_FAILURE;
    }
    
    // There will be a compute-derived latency hit 
    // when the number of warps is greater than 
    // max_warps_per_sm * num_sm. (64 is max warps per sm core)
    // num_warps = ceil(num_blocks * num_threads_per_block / warpsize)
    // = ceil(N * M / warpsize) <= sm_cores * max_warps_per_sm_core
    for (int N = 1; N <= 4 * warps_at_max_occupancy; N++) {
        int M = THREADS_PER_BLOCK;
        launch_kernel(kernel, N, M, f_out);
    }
    
    // k_dummy<<<N, M>>>();
    // printf("%d\n", i);
    fprintf(f_out, "NumBlocks,ThreadsPerBlock,Latency(ms)\n");
    fclose(f_out);

    return EXIT_SUCCESS;
}


