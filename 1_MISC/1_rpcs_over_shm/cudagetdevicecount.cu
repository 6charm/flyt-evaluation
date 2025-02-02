/*
 * Test rpcs in shm mode: cudagetdevicecount()
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define NUM_ITER 1000

int main() {
    int gpu_n;
    for (int i = 0; i < NUM_ITER; i++) {
        cudaGetDeviceCount(&gpu_n);
    }
    printf("CUDA-capable device count: %i\n", gpu_n);
}