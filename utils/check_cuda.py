# check_cuda.py
import torch

print("="*50)
print("CUDA / GPU INFORMATION")
print("="*50)

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
else:
    print("GPU: None (CUDA not available)")
    
    # Cek apakah PyTorch diinstall dengan CUDA support
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version (build): {torch.version.cuda if torch.version.cuda else 'None (CPU build)'}")