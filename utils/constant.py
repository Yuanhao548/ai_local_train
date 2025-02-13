import os
from pathlib import Path

import torch

# 获取当前文件的路径对象
current_file_path = Path(__file__)
# 获取当前文件的父目录
root_path = current_file_path.parent.parent

BASE_MODEL_NAME = 'Deepseek_R1_Distill_Qwen_7B'
BASE_MODEL_DIR = Path(os.path.join(root_path, BASE_MODEL_NAME))

BASE_MODEL_NAME_OR_PATH = Path(os.path.join(BASE_MODEL_DIR, "deepseek-ai", BASE_MODEL_NAME))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

TRAIN_DATA_SET_PATH = Path(os.path.join(root_path, "static", "train_datas_set.json"))  # 训练数据集路径
VALID_DATA_SET_PATH = Path(os.path.join(root_path, "static", "validation_datas_set.json"))  # 验证数据集路径

TRAINED_LORA_WEIGHTS_MODEL_DIR = Path(os.path.join(root_path, "Lora_Weights_Model"))  # LoRA权重保存路径

