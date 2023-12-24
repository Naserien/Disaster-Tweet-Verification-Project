import torch

# 打印PyTorch的版本号
print(torch.__version__)

# 打印是否支持CUDA（GPU加速）
print(f"CUDA available: {torch.cuda.is_available()}")