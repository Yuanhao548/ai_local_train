import gc
import re
from functools import lru_cache

import torch

from utils.constant import DEVICE
from utils.exception import FileEmptyError


def file_empty_to_exception(f, ex_info):
    content = f.read()
    if not content:
        raise FileEmptyError(ex_info)
    return content


# 清理 设备 内存函数
def torch_gc(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc_collect()


def gc_collect():
    gc.collect()


# 缓存分词器
@lru_cache(maxsize=10)
def cached_tokenize(tokenizer, text, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,  # 自动填充
        return_attention_mask=True  # 生成attention_mask
    )
    # 将 input_ids 和 attention_mask 移动到与模型相同的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def generate_ai_response(inputs, model, pad_token_id, tokenizer, device=DEVICE):
    # 生成时传入必要参数
    generated_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # 传入attention_mask
        pad_token_id=pad_token_id,  # 传入 pad_token_id
        max_new_tokens=50,          # 控制生成的最大长度
        num_beams=1,                # 禁用束搜索
        repetition_penalty=1.2,    # 避免重复生成
        do_sample=True,             # 启用采样
        temperature=0.8,            # 控制采样随机性
        top_p=0.9,                  # 核采样
        use_cache=True             # 启用缓存以加速生成
        # 其他生成参数...
    )

    # 提取生成的 ID（跳过输入部分）
    input_length = inputs["input_ids"].shape[1]  # 输入序列的长度
    generated_ids = generated_ids[:, input_length:]  # 截取生成部分

    # 解码生成的 ID
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    del generated_ids, inputs
    torch_gc(device)
    return response


# 文本分割函数
def split_resp_think_text(text):
    pattern = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL)  # 定义正则表达式模式
    match = pattern.search(text)  # 匹配 <think>思考过程</think>回答

    if match:  # 如果匹配到思考过程
        think_content = match.group(1).strip()  # 获取思考过程
        answer_content = match.group(2).strip()  # 获取回答
    else:
        think_content = ""  # 如果没有匹配到思考过程，则设置为空字符串
        answer_content = text.strip()  # 直接返回回答

    return think_content, answer_content