# CUDA Multi-Process Service (MPS)
`getent passwd` : List all users with UID.

#### Start MPS control daemon. When a new CUDA app is run, if system clean, control daemon will spawn a new MPS server instance for the $UID
that started the control daemon. 
```
As $UID, run the commands:

export CUDA_VISIBLE_DEVICES=0 # Select GPU 0.

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Select a location that's accessible to the given $UID. Used by CUDA app to communicate with daemon.

export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Select a location that's accessible to the given $UID

nvidia-cuda-mps-control -d # Start the daemon.
```
This will start the MPS control daemon that will spawn a new MPS Server instance for that $UID starting an application and associate it with GPU visible to the control daemon. 

`echo quit | nvidia-cuda-mps-control` to stop the daemon. The control.log file will be updated with exit status. 

#### Monitor the mps UDS
UDS Path: `/tmp/nvidia-mps/control`

#### GPU Compute Modes
`nvidia-smi -i 2 -c EXCLUSIVE_PROCESS` sets GPU index -i to compute mode -c

```
Three Compute Modes are supported via settings accessible in nvidia-smi:

PROHIBITED – the GPU is not available for compute applications.

EXCLUSIVE_PROCESS — the GPU is assigned to only one process at a time, and individual process threads may submit work to the GPU concurrently.

DEFAULT – multiple processes can use the GPU simultaneously. Individual threads of each process may submit work to the GPU simultaneously.

Using MPS effectively causes EXCLUSIVE_PROCESS mode to behave like DEFAULT mode for all MPS clients. MPS will always allow multiple clients to use the GPU via the MPS server.

When using MPS it is recommended to use EXCLUSIVE_PROCESS mode to ensure that only a single MPS server is using the GPU, which provides additional insurance that the MPS server is the single point of arbitration between all CUDA processes for that GPU
```

# MPS Resource Limits
`CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING`
"When non-uniform partitioning capability is enabled in an MPS client’s environment, client CUDA contexts can have different active thread percentages within the same client process via setting CUDA_MPS_ACTIVE_THREAD_PERCENTAGE before context creations. The device attribute cudaDevAttrMultiProcessorCount will reflect the active thread percentage and return the portion of available SMs that can be used by the client CUDA context current to the calling thread."

# Experimental results
Using nbody, with MPS, no resource limits. GPU at 100%util with 6proc, 16384 bodies
`./nbody --benchmark --fp64` with EXCLUSIVE_PROCESS : 900ms per process.

`./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 &` with EXCLUSIVE_PROCESS : 5380ms per process 

`./nbody --benchmark --fp64` with DEFAULT : 900ms per process.

`./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 &` with DEFAULT: 5380ms per process, each process runs indep on nvidia-smi

Conclude: 
a. With MPS, no difference in exec between DEFAULT and EXCLUSIVE_PROCESS. But why???

b. nvidia-smi shows individual processes on GPU, not one large MPS process. (in fact its a small mps server that doesnt change size)


Using nbody, without MPS GPU at 100%util with 6 proc

`./nbody --benchmark --fp64` with DEFAULT: 900ms per process

`./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 &` with DEFAULT: 5700ms per process (more than MPS bec of timeslice scheduling of multiple process contexts.)


`./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 & ./nbody --benchmark --fp64 &` with EXCLUSIVE_PROCESS: cudaSetDevice fails for parallel processes when EXCLUSIVE_PROCESS is set without MPS.

*On A4000, the latencies drop to 0.5 x latency of GeForce 1650Ti*




