import json

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from trained.terminology import terminology_dict
from utils.conf import file_empty_to_exception, torch_gc, gc_collect
from utils.constant import BASE_MODEL_NAME_OR_PATH, DEVICE, TRAIN_DATA_SET_PATH, TRAINED_LORA_WEIGHTS_MODEL_DIR, \
    TRAINING_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE, TRAINING_ARGS_GRADIENT_ACCUMULATION_STEPS, LORA_CONFIG_R, \
    LORA_CONFIG_LORA_ALPHA, TORCH_DTYPE, LORA_CONFIG_TASK_TYPE, IS_HIGH_PERF, TRAIN_MODEL_LOAD_IN_8BIT
from utils.exception import FileEmptyError

device_map = {"": DEVICE}
train_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_OR_PATH, use_fast=True)
train_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME_OR_PATH,
    torch_dtype=eval(TORCH_DTYPE),  # 使用FP16精度
    low_cpu_mem_usage=True,     # 显示启用低内存加载模式
    device_map=device_map if IS_HIGH_PERF else {"": "cpu"},  # device_map 字典来指定模型加载时各个模块的设备映射
    # load_in_8bit=TRAIN_MODEL_LOAD_IN_8BIT,  # 使用 8 比特加载
)

# ---------- 数据处理 ----------
class TestCaseDataProcessor:
    """
    测试用例数据处理器
    训练数据结构示例：
    [
        {
            "requirement": "用户登录功能需验证用户名和密码",
            "test_points": ["有效输入验证", "无效用户名处理", "密码错误处理"],
            "test_cases": [
                "等价类划分：使用有效用户名和正确密码",
                "边界值分析：用户名长度超限测试",
                "错误推测：特殊字符密码测试"
            ]
        }
    ]
    """

    def __init__(self, tokenizer, terminology_dict):
        self.tokenizer = tokenizer
        self.terminology_dict = terminology_dict
        self.max_length = 512  # 根据显存调整

    def normalize_terms(self, text):
        """将文本中的术语标准化"""
        for term in self.terminology_dict.get_all_terms():
            if term in text:
                text = text.replace(term, f"[{term}]")  # 标记术语
        return text

    def format_prompt(self, sample):
        """通用的构造指令微调格式，并标准化术语"""
        requirement = self.normalize_terms(sample['requirement'])
        test_points = [self.normalize_terms(point) for point in sample['test_points']]

        prompt = f"""根据以下需求生成测试用例：
需求：{requirement}
测试点分析：{', '.join(test_points)}
生成的测试用例（使用多种测试方法）：\n"""

        completion = "\n".join(sample['test_cases']) + self.tokenizer.eos_token
        return prompt + completion

    def process_data(self, data):
        """数据处理流水线"""
        formatted_data = [self.format_prompt(item) for item in data]
        return self.tokenizer(
            formatted_data,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )


# ---------- LoRA配置 ----------
def setup_lora(model):
    """配置LoRA参数"""
    config = LoraConfig(
        r=LORA_CONFIG_R,  # LoRA秩
        lora_alpha=LORA_CONFIG_LORA_ALPHA,  # 缩放系数
        target_modules=["q_proj", "v_proj"],  # 目标模块
        lora_dropout=0.05,
        bias="none",
        task_type=LORA_CONFIG_TASK_TYPE,
    )
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, config)


def augment_data_with_synonyms(data, terminology_dict):
    """使用术语的同义词扩展训练数据"""
    augmented_data = []
    for item in data:
        # 原始数据
        augmented_data.append(item)

        # 使用同义词生成新数据
        new_item = item.copy()
        for term in terminology_dict.get_all_terms():
            if term in new_item["requirement"]:
                synonyms = terminology_dict.get_synonyms(term)
                if synonyms:
                    new_item["requirement"] = new_item["requirement"].replace(term, synonyms[0])
                    augmented_data.append(new_item)
    return augmented_data


# ---------- 训练函数 ----------
def train():
    # 初始化基座模型
    try:
        # 将模型移动到指定设备
        train_model.to(DEVICE)

        torch_gc(DEVICE)

        # 准备训练数据
        with open(TRAIN_DATA_SET_PATH) as f:
            content = file_empty_to_exception(f, "待训练数据为空")
            raw_data = json.loads(content)
        processor = TestCaseDataProcessor(train_tokenizer, terminology_dict)

        # 数据增强
        augmented_data = augment_data_with_synonyms(raw_data, terminology_dict)

        processed_data = processor.process_data(augmented_data)
        # dataset = load_dataset('json', data_files=str(TRAIN_DATA_SET_PATH), streaming=True)    # streaming 使用流式加载
        dataset = load_dataset('json', data_files=str(TRAIN_DATA_SET_PATH))['train']


        # 释放不再使用的变量
        del raw_data, augmented_data
        gc_collect()

        # 配置LoRA
        model = setup_lora(train_model)
        model.print_trainable_parameters()  # 显示可训练参数

        # 计算 max_steps
        dataset_size = len(dataset)
        per_device_train_batch_size = TRAINING_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE
        gradient_accumulation_steps = TRAINING_ARGS_GRADIENT_ACCUMULATION_STEPS
        num_train_epochs = 3

        max_steps = (dataset_size // (per_device_train_batch_size * gradient_accumulation_steps)) * num_train_epochs

        # 训练参数设置
        training_args = TrainingArguments(
            output_dir=TRAINED_LORA_WEIGHTS_MODEL_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=TRAINING_ARGS_PER_DEVICE_TRAIN_BATCH_SIZE,  # 根据显存调整
            gradient_accumulation_steps=TRAINING_ARGS_GRADIENT_ACCUMULATION_STEPS,
            learning_rate=2e-5,
            fp16=False,
            logging_steps=10,
            save_strategy="epoch",
            remove_unused_columns=False,  # 当该参数为 True 时，Trainer 会移除数据集中模型前向传播方法不需要的列。设置为 False 可以避免移除这些列
            max_steps=max_steps,
        )

        # 定义 data_collator 函数
        def data_collator(data):
            input_ids = []
            attention_mask = []
            labels = []
            for sample in data:
                # 直接处理 sample，假设 sample 已经是处理好的格式
                formatted_sample = processor.format_prompt(sample)
                tokenized_sample = processor.tokenizer(
                    formatted_sample,
                    max_length=processor.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids.append(tokenized_sample['input_ids'])
                attention_mask.append(tokenized_sample['attention_mask'])
                labels.append(tokenized_sample['input_ids'])
            # 将列表转换为张量
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            labels = torch.cat(labels, dim=0)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

        # 开始训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        trainer.train()
        model.save_pretrained(TRAINED_LORA_WEIGHTS_MODEL_DIR)
        print("训练完成，LoRA 权重已保存。")
        return {"message": "训练完成，LoRA 权重已保存。", "status": 200}
    except FileEmptyError as e:
        print(e)
        return {"message": e, "status": 500}
    except Exception as e:
        print(e)
        return {"message": f"训练过程中出现错误: {e}", "status": 500}


if __name__ == "__main__":
    train()