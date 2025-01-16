#!/bin/bash

# -- Used by Flyt to start the MPS daemon
export CUDA_VISIBLE_DEVICES="0"

# Flyt RPC server starts with `CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING = 1`
# ---> This allows non-uniform SM core reservation for cuda contexts created by the same MPS client
# (CUDA process). Multiple contexts are often used when the cuda process is multithreaded. But its
# likely unnecessary. 
# 
# In the standard mode without multiprocessor partitioning, the MPS server can handle several 
# unique CUDA processes (clients), each with unique sm core allocations (no restriction), 
# assuming they all use a single context.
# ---> If more contexts were created, they would be given the same SM core allocation as the 
# original context.
#
# in Volta MPS, multiple cuContexts from a single cuda process with individual SM core reservations are 
# coalesced into a single MPS context to be scheduled onto the GPU. (No time slicing)
# ===
# EXCLUSIVE_PROCESS GPU compute mode ensures that only
# a single process may be active on the GPU at once.
# ---> When using MPS, we want to ensure that a single MPS server 
# has exclusive access to the GPU. 
# ===
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # set GPU index -i to compute mode -c
nvidia-cuda-mps-control -d
