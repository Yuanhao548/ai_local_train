# import gc
# import os
# from functools import lru_cache
#
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# # from utils.snapshot_download import model_dir
#
# # 选择合适的设备
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# device_map = {"": DEVICE}
#
# # 加载模型和分词器
# model_name_or_path = os.path.join(model_dir, "deepseek-ai", "DeepSeek-R1-Distill-Qwen-7B")
#
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     torch_dtype=torch.float16,  # 使用FP16精度
#     low_cpu_mem_usage=True,     # 显示启用低内存加载模式
#     device_map=device_map
# )
#
#
# # 清理 GPU 内存函数
# def torch_gc():
#     if torch.backends.mps.is_available():
#         try:
#             from torch.mps import empty_cache
#             empty_cache()
#         except AttributeError:
#             pass
#     gc.collect()
#
#
# # 缓存分词器
# @lru_cache(maxsize=10)
# def cached_tokenize(text):
#     return tokenizer(
#         text,
#         return_tensors="pt",
#         padding=True,  # 自动填充
#         return_attention_mask=True  # 生成attention_mask
#     )
#
#
# # 调用模型处理自然语言
# def call_deepseek_model(messages):
#     try:
#         # 检查并设置pad_token
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为pad_token
#
#         # 确认pad_token_id
#         pad_token_id = tokenizer.pad_token_id
#
#         # 调用模型进行对话生成
#         input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#
#         # 处理输入时启用padding并返回attention_mask
#         model_inputs = cached_tokenize(input_text)
#
#         # 将 input_ids 和 attention_mask 移动到与模型相同的设备上
#         model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
#
#         # 生成时传入必要参数
#         generated_ids = model.generate(
#             model_inputs["input_ids"],
#             attention_mask=model_inputs["attention_mask"],  # 传入attention_mask
#             pad_token_id=pad_token_id,  # 传入 pad_token_id
#             max_new_tokens=50,          # 控制生成的最大长度
#             num_beams=1,                # 禁用束搜索
#             repetition_penalty=1.2,    # 避免重复生成
#             do_sample=True,             # 启用采样
#             temperature=0.8,            # 控制采样随机性
#             top_p=0.9,                  # 核采样
#             use_cache=True             # 启用缓存以加速生成
#             # 其他生成参数...
#         )
#
#         # 提取生成的 ID（跳过输入部分）
#         input_length = model_inputs["input_ids"].shape[1]  # 输入序列的长度
#         generated_ids = generated_ids[:, input_length:]  # 截取生成部分
#
#         # 解码生成的 ID
#         response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#         del generated_ids, model_inputs
#         torch_gc()
#         return response
#     except RuntimeError as e:
#         print(f"Runtime error during model generation: {e}")
#         return None
#     except ValueError as e:
#         print(f"Value error during model generation: {e}")
#         return None
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return None
