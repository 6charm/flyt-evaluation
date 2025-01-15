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