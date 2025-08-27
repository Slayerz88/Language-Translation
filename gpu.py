import torch

# Check if CUDA (GPU) is available
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU Index:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Memory Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Memory Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")
