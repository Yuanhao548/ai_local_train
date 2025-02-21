import os
from pathlib import Path

import torch

# 获取当前文件的路径对象
current_file_path = Path(__file__)
# 获取当前文件的父目录
root_path = current_file_path.parent.parent

BASE_MODEL_NAME = 'DeepSeek-R1-Distill-Qwen-1.5B'
# BASE_MODEL_NAME = 'DeepSeek-R1-Distill-Qwen-7B'
BASE_MODEL_DIR = Path(os.path.join(root_path, BASE_MODEL_NAME))

CORPUS_DIR_PATH = Path(os.path.join(root_path, "Corpus", "ai_yuliao"))
BASE_MODEL_NAME_OR_PATH = Path(os.path.join(BASE_MODEL_DIR, "deepseek-ai", BASE_MODEL_NAME))
SENTENCE_EMBEDDING_MODEL_PATH = Path(os.path.join(BASE_MODEL_DIR, "embedding_model"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# DEVICE = torch.device("cpu")
DEVICE_MAP = {"": DEVICE}

TRAIN_DATA_SET_PATH = Path(os.path.join(root_path, "static", "train_datas_set.json"))  # 训练数据集路径
VALID_DATA_SET_PATH = Path(os.path.join(root_path, "static", "validation_datas_set.json"))  # 验证数据集路径
TERMINOLOGY_DICTIONARY_PATH = Path(os.path.join(root_path, "static", "terminology_dictionary.json"))  # 专业领域词典

TRAINED_LORA_WEIGHTS_MODEL_DIR = Path(os.path.join(root_path, "Lora_Weights_Model"))  # LoRA权重保存路径


# 调整模型的性能参数
IS_HIGH_PERF = 1

# training_args
TRAINING_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE = 1 if not IS_HIGH_PERF else 4
TRAINING_ARGS_GRADIENT_ACCUMULATION_STEPS = 1 if not IS_HIGH_PERF else 8  # 梯度累积步数降为1

# train_model
TRAIN_MODEL_LOAD_IN_8BIT = False if IS_HIGH_PERF else True

# LoraConfig
LORA_CONFIG_R = 4 if not IS_HIGH_PERF else 8
LORA_CONFIG_LORA_ALPHA = 16 if not IS_HIGH_PERF else 32
LORA_CONFIG_TASK_TYPE = "CAUSAL_LM"   # SEQ_CLS、SEQ_2_SEQ_LM、CAUSAL_LM、TOKEN_CLS、QUESTION_ANS、FEATURE_EXTRACTION

TORCH_DTYPE = "torch.float32"   # 使用 FP32 避免量化开销
