import torch
import platform
import os

def get_best_device() -> torch.device:
    """
    Get the best available device with preference for:
    1. NVIDIA RTX GPU on Windows/Linux
    2. Apple Silicon GPU on macOS
    3. CPU as fallback
    """
    if torch.cuda.is_available():
        # Print all available CUDA devices
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s):")
        
        # Look for NVIDIA RTX specifically
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {device_name}")
            
            # Prefer NVIDIA RTX GPU if available
            if "RTX" in device_name or "NVIDIA" in device_name:
                selected_device = torch.device(f"cuda:{i}")
                print(f"Selected NVIDIA GPU: {device_name}")
                return selected_device
        
        # If no RTX found, use the first CUDA device
        print(f"No NVIDIA RTX GPU found, using default CUDA device")
        return torch.device("cuda:0")
    
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Apple Silicon optimizations
        print("Using Apple Silicon GPU with MPS")
        
        # Set environment variables to optimize MPS performance
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallbacks for unsupported ops
        
        # Check for macOS 13+ for best MPS performance
        mac_version = platform.mac_ver()[0]
        if mac_version:
            major_version = int(mac_version.split('.')[0])
            if major_version < 13:
                print(f"Warning: macOS {mac_version} detected. MPS performs best on macOS 13+")
        
        return torch.device("mps")
    
    else:
        # CPU fallback with thread optimization
        print("No GPU found, using CPU")
        
        # Set optimal thread settings for CPU
        cpu_count = os.cpu_count()
        if cpu_count:
            # Use most cores but leave some for system
            optimal_threads = max(1, cpu_count - 2)
            torch.set_num_threads(optimal_threads)
            print(f"Set PyTorch to use {optimal_threads} of {cpu_count} CPU threads")
            
        return torch.device("cpu")