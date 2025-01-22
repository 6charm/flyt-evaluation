# FILE: check_mps_avail.py
# ------------------------
# Refer https://www.blopig.com/blog/2024/11/big-compute-doesnt-want-you-to-know-this-maximising-gpu-usage-with-cuda-mps/#:~:text=CUDA%20Multi%2DProcess%20Service%20(MPS,GPU%20utilization%20and%20higher%20throughput.
import subprocess
import os

def mps_control_exited(file_path, target_message="Exit with status 0"):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines and target_message in lines[-1]:
            return True
        return False

def check_cuda_mps_status():
    """Check if CUDA MPS is properly configured and running."""
    try:
        # Verify CUDA availability
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)

        # Check available GPUs
        gpu_idx = os.getenv('CUDA_VISIBLE_DEVICES')
        if not gpu_idx:
            print("Warning: CUDA_VISIBLE_DEVICES not set")
            return False
        
        # Check MPS environment variable
        mps_pipe = os.getenv('CUDA_MPS_PIPE_DIRECTORY')
        if not mps_pipe:
            print("Warning: CUDA_MPS_PIPE_DIRECTORY not set")
            return False
            
        # Verify MPS control file existence.
        # /tmp/nvidia-mps/control is a socket for user to 
        # communicate with control daemon. 
        control_file = os.path.join(mps_pipe, "control")
        if not os.path.exists(control_file):
            print("Warning: MPS control file not found")
            return False
        
        # Check MPS logs
        mps_log = os.getenv('CUDA_MPS_LOG_DIRECTORY')
        control_log = os.path.join(mps_log, "control.log")
        if not os.path.exists(control_log):
            print("Warning: MPS control log not found")
            return False
        
        if not mps_control_exited(control_log):
            print("All checks passed, control daemon is running")
        else:
            print("Control daemon not running.")
        
        return True
        
    except subprocess.CalledProcessError:
        print("Warning: Unable to verify CUDA setup")
        return False

check_cuda_mps_status()