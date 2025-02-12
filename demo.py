import torch

# 检查当前内存使用
print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
