/**
 * FILE: multi_ctx.cu
 * ------------------
 * To test the capabilities of CUDA MPS resource provisioning by comparing kernel execution
 * latencies.
 * Workload: Long running kernel with 100% SM utilization without MPS SM limits. 
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

int main(int argc,  char *argv[]) {
    
}