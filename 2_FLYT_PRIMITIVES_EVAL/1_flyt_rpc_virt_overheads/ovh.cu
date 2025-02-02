/**
 * FILE: test-rpc-virt-ovh.cu
 * --------------------------
 * Based on command-line option, do cugetdeivce 1000 iterations
 * cudalaunchkernel with varying number of parameters.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define NUM_ITER 1000  // Number of iterations to run

__global__ void dummy_kernel(int param) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("Thread %d received param: %d\n", idx, param);
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

        // printf("Device %d: %s\n", device_id, device_prop.name);
    }
}

// Define kernels with a literal number of parameters
#define DEFINE_KERNEL(num_params)                                               \
__global__ void dummy_kernel_##num_params(int p1, int p2, ##__VA_ARGS__) {      \
    printf("Kernel with " #num_params " parameters launched.\n");               \
    for (int i = 1; i <= num_params; i++) {                                     \
        printf("Thread %d, param[%d] = %d\n", threadIdx.x, i, i);               \
    }                                                                           \
}


// Kernel for 4 parameters
__global__ void dummy_kernel_4(int p1, int p2, int p3, int p4) {
    printf("Kernel with 4 parameters: p1 = %d, p2 = %d, p3 = %d, p4 = %d\n", p1, p2, p3, p4);
}

// Kernel for 8 parameters
__global__ void dummy_kernel_8(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8) {
    printf("Kernel with 8 parameters: p1 = %d, p2 = %d, p3 = %d, p4 = %d, p5 = %d, p6 = %d, p7 = %d, p8 = %d\n",
           p1, p2, p3, p4, p5, p6, p7, p8);
}

// Kernel for 16 parameters
__global__ void dummy_kernel_16(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8, 
                                 int p9, int p10, int p11, int p12, int p13, int p14, int p15, int p16) {
    printf("Kernel with 16 parameters: p1 = %d, p2 = %d, p3 = %d, p4 = %d, p5 = %d, p6 = %d, p7 = %d, p8 = %d, "
           "p9 = %d, p10 = %d, p11 = %d, p12 = %d, p13 = %d, p14 = %d, p15 = %d, p16 = %d\n", 
           p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
}


void run_launch_kernel(int num_params) {
    for (int i = 1; i <= NUM_ITER; i++) {
        // Print out the kernel launch for each iteration
        // printf("Launching kernel with %d parameters.\n", num_params);

        // Launch the appropriate kernel based on num_params using switch-case
        switch (num_params) {
            case 4:
                dummy_kernel_4<<<1, 1>>>(1, 2, 3, 4);  // Launch with 4 parameters
                break;
            case 8:
                dummy_kernel_8<<<1, 1>>>(1, 2, 3, 4, 5, 6, 7, 8);  // Launch with 8 parameters
                break;
            case 16:
                dummy_kernel_16<<<1, 1>>>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);  // Launch with 16 parameters
                break;
            // Add more cases as needed for different parameter counts
            default:
                printf("Unsupported parameter count: %d\n", num_params);
                break;
        }

        // Synchronize and check for errors
        cudaError_t err = cudaDeviceSynchronize();
        check_cuda_error(err);
    }
}

void *get_random_array(long long size) {
    char *arr = (char*)malloc(size);

    if (arr == NULL) {
        fprintf(stderr, "Unable to allocate array on cpu\n");
        return NULL;
    }

    for (long long i = 0; i < size; i++) {
        arr[i] = (char)(rand() % CHAR_MAX);
    }
    return arr;
}


void run_memcpy(size_t array_sz) {
    long *h_a = (long *)get_random_array(array_sz);
    long *d_a;

    cudaMalloc(&d_a, array_sz);
    for (int i = 0; i < NUM_ITER; i++) {
        cudaMemcpy(d_a, h_a, array_sz, cudaMemcpyHostToDevice);
    }
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 2) {
        fprintf(stderr, "Usage: %s <cuda-call-type> [num_params]\n", argv[0]);
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
            run_launch_kernel(atoi(argv[2])); // 4 8 16
            break;
        case 3: 
            run_memcpy(atoi(argv[2]));
        default:
            fprintf(stderr, "Invalid command-line argument: %d. Use '1' for cuGetDevice or '2' for kernel launch.\n", cuda_call_type);
            return -1;
    }

    return 0;
}
