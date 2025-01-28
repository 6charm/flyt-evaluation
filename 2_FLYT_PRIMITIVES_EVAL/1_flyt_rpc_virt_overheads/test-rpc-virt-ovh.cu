/**
 * FILE: test-rpc-virt-ovh.cu
 * --------------------------
 * Based on command-line option, do cugetdeivce 1000 iterations
 * cudalaunchkernel with varying number of parameters.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_ITER 1000  // Number of iterations to run

__global__ void dummy_kernel(int param) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread %d received param: %d\n", idx, param);
}

void check_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void run_get_device() {
    for (int i = 0; i < NUM_ITER; i++) {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        check_cuda_error(err);

        if (device_count == 0) {
            fprintf(stderr, "No CUDA-capable devices found!\n");
            return;
        }

        int device_id = 0;  // Select the first CUDA device (device 0)
        err = cudaSetDevice(device_id);
        check_cuda_error(err);

        // Output device information for verification
        cudaDeviceProp device_prop;
        err = cudaGetDeviceProperties(&device_prop, device_id);
        check_cuda_error(err);

        printf("Device %d: %s\n", device_id, device_prop.name);
    }
}

void run_launch_kernel() {
    for (int i = 0; i < NUM_ITER; i++) {
        int num_params = (i % 5) + 1; // Vary the number of parameters for the kernel
        printf("Launching kernel with %d parameter(s).\n", num_params);

        // Launch the kernel with `num_params` threads
        dummy_kernel<<<1, num_params>>>(num_params);
        cudaError_t err = cudaDeviceSynchronize();
        check_cuda_error(err);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <cuda-call-type>\n", argv[0]);
        fprintf(stderr, "Where <cuda-call-type> can be '1' for cuGetDevice or '2' for kernel launch.\n");
        return -1;
    }

    // Command-line argument specifies which CUDA call to make
    int cuda_call_type = atoi(argv[1]);

    // Call the appropriate function based on the argument
    switch (cuda_call_type) {
        case 1:
            run_get_device();
            break;
        case 2:
            run_launch_kernel();
            break;
        default:
            fprintf(stderr, "Invalid command-line argument: %d. Use '1' for cuGetDevice or '2' for kernel launch.\n", cuda_call_type);
            return -1;
    }

    return 0;
}
