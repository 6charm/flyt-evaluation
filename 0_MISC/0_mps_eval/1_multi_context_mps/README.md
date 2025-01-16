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