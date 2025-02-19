import gc
import os
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from trained.validate import validator
from utils.conf import torch_gc, cached_tokenize, generate_ai_response
from utils.constant import BASE_MODEL_NAME_OR_PATH, DEVICE

tokenizer = validator.tokenizer
model = validator.model
pad_token_id = validator.tokenizer_pad_token_id


# 调用模型处理自然语言
def call_lora_model(messages):
    try:
        # 调用模型进行对话生成
        input_text = validator.get_input_text(messages)

        inputs = cached_tokenize(tokenizer, input_text, DEVICE)
        response = generate_ai_response(inputs, model, pad_token_id, tokenizer)
        return response
    except RuntimeError as e:
        print(f"Runtime error during model generation: {e}")
        return None
    except ValueError as e:
        print(f"Value error during model generation: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
