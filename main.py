import multiprocessing
import os
import sys

# Configure CUDA library paths from the virtual environment
venv_site_packages = os.path.join(os.getcwd(), ".venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
nvidia_base = os.path.join(venv_site_packages, "nvidia")

if os.path.exists(nvidia_base):
    cuda_libs = [
        os.path.join(nvidia_base, "cuda_runtime", "lib"),
        os.path.join(nvidia_base, "cublas", "lib"),
        os.path.join(nvidia_base, "cusparse", "lib"),
        os.path.join(nvidia_base, "nvjitlink", "lib"),
        os.path.join(nvidia_base, "cuda_nvrtc", "lib"),
        os.path.join(nvidia_base, "cudnn", "lib"),
        os.path.join(nvidia_base, "cufft", "lib"),
        os.path.join(nvidia_base, "cuda_cupti", "lib"),
        os.path.join(nvidia_base, "cusolver", "lib"),
    ]
    
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld_path = ":".join(cuda_libs)
    if current_ld_path:
        new_ld_path = f"{new_ld_path}:{current_ld_path}"
    
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    # Also set TF_CPP_MIN_LOG_LEVEL to 0 to see if it still complains about missing libs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

from master_process import MasterProcess

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    master = MasterProcess()
    master.train_agents()
