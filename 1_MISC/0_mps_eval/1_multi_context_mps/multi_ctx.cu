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
#include <curand.h>
#include <stdlib.h>

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
    return 0;
}

// A[M*K] X B[K*N] = C'[M*N]
// C = α*C'+β*C
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x; // index of 
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      // perform a row-column dot product. 
      // In matrix A, K is the length of a Row (# columns). 
      // the first multiplicand refers to an element of the
      // row being handled by thread with global id x. A key 
      // idea is that the matrix is stored in Row-Major form, 
      // therefore the ~strided access pattern.
      // --
      // The second refers to the element of the
      // corresponding column of B that is required
      // according to matrix multiplication semantics.
      // --
      // Each GPU thread (remember a kernel defintion specifies the
      // work of a single thread) computes the row * col dot product
      // for the row and column it is indexed to handle.
      tmp += A[x * K + i] * B[i * N + y]; 
    }
	  // after the loop, tmp is the element C[x][y], of matrix C
    // `C[x * N + y]` refers to the element in the (x-1)th row and (y-1)th column
    // of the matrix C -> Which is the one this thread is computing!
    // The below line updates the value of that element according to 
    // the SGEMM formula: C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

#define CEIL_DIV(a, b) ((a % b) != 0) ? (a / b + 1) : (a / b)

typedef void(*kernel_func_ptr)(int M, int N, int K, float alpha, const float *A_dev,
                            const float *B_dev, float beta, float *C_dev);

void launch_kernel(FILE *f_out, int M, int N, int K, float alpha, const float *A_dev,
                            const float *B_dev, float beta, float *C_dev) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create as many blocks as necessary to map all of C
    // dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 gridDim(32, 32, 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32, 1);

    cudaEventRecord(start);
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A_dev, B_dev, beta, C_dev);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float t_ms = 0;
    cudaEventElapsedTime(&t_ms, start, stop);
    printf("here1\n");
    printf("(%d, 1024): %.4f ms\n", N, t_ms);
    // fprintf(f_out, "%d,%.4f\n", N,t_ms);
    printf("here2\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
}


// A 25:75 SM alloc vs a 50:50 SM alloc
// The light kernel needs to use [0%-25%] of SMs to meet the SLA.
// The heavy kernel needs to use [50%-75%] of its SMs to meet the SLA.
// ---> Latency SLA: The kernel exection time with the above SM reservations.
// When the sms are changed (25->50 and 75->50, the lighter kernel should be unaffected, 
// but the heavier kernel will slow down.)
void derive_SLA(kernel_func_ptr w_kernel) {


}

void fill_matrix(float *mat_d, int n_rows, int n_cols) {
    float *mat_h = (float *)malloc(n_cols *n_rows* sizeof(float));
    mat_h[0] = 1.4;
    for (int i = 1; i < n_cols *n_rows; i++) {
        float tmp = mat_h[i-1];
        mat_h[i] = (int)((tmp* 13 + i)) % 1024;
    }
    cudaMemcpy(mat_d, mat_h, n_cols *n_rows * sizeof(float), cudaMemcpyHostToDevice);
}

#define THREAD_COUNT 2
#define alpha 0.2
#define beta 0.7

int main(int argc,  char *argv[]) {
    derive_SLA(sgemm_naive);
    if (argc != 4) {
        printf("Usage: ./mps-bench <rows_A> <cols_A/rows_B> <cols_B>\n");
        exit(EXIT_FAILURE);
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    float *A_dev, *B_dev, *C_dev;
    cudaMalloc((void **)&A_dev, M * K);
    fill_matrix(A_dev, M, K);

    cudaMalloc((void **)&B_dev, K * N);
    fill_matrix(B_dev, K, N);

    float *C_host = (float *)malloc(sizeof(float) * M * N);
    cudaMalloc((void **)&C_dev, M * N);

    char f_name[128];
    sprintf(f_name,"mps_bench_matmul_%d_.csv", 1);
    printf("%s\n", f_name);
    FILE *f_out = fopen(f_name, "w");
    launch_kernel(f_out, M, N, K, alpha, A_dev, B_dev, beta, C_dev);
    printf("here\n");

    cudaMemcpy(C_host, C_dev, M*N, cudaMemcpyDeviceToHost);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    free(C_host);
}