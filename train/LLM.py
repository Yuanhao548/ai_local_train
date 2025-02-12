from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class DeepSeek_R1_Distill_Qwen_LLM(LLM):
    # 基于本地 DeepSeek_R1_Distill_Qwen 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        try:
            # 选择合适的设备
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
            self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
            self.model = self.model.to(device)  # 将模型移动到合适的设备上
            print("完成本地模型的加载")
        except Exception as e:
            print(f"模型加载失败: {e}")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        try:
            # 选择合适的设备
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(device)
            generated_ids = self.model.generate(model_inputs.input_ids, attention_mask=model_inputs['attention_mask'], max_new_tokens=8192)
            generated_ids = [
                output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 释放不再使用的变量和缓存
            del model_inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                try:
                    from torch.mps import empty_cache
                    empty_cache()
                except AttributeError:
                    pass

            return response
        except Exception as e:
            print(f"文本生成失败: {e}")
            return None

    @property
    def _llm_type(self) -> str:
        return "DeepSeek_R1_Distill_Qwen_LLM"