import json

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from train.terminology_dict import terminology_dict
from utils.constant import BASE_MODEL_NAME_OR_PATH, DEVICE, TRAIN_DATA_SET_PATH, TRAINED_LORA_WEIGHTS_MODEL_DIR

train_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_OR_PATH)
train_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME_OR_PATH)


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
        r=8,  # LoRA秩
        lora_alpha=32,  # 缩放系数
        target_modules=["q_proj", "v_proj"],  # 目标模块
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
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
        train_model.to(DEVICE)

        # 准备训练数据
        with open(TRAIN_DATA_SET_PATH) as f:
            raw_data = json.load(f)
        processor = TestCaseDataProcessor(train_tokenizer, terminology_dict)

        # 数据增强
        augmented_data = augment_data_with_synonyms(raw_data, terminology_dict)

        processed_data = processor.process_data(augmented_data)
        dataset = load_dataset('json', data_files=TRAIN_DATA_SET_PATH)['train']

        # 配置LoRA
        model = setup_lora(train_model)
        model.print_trainable_parameters()  # 显示可训练参数

        # 训练参数设置
        training_args = TrainingArguments(
            output_dir=TRAINED_LORA_WEIGHTS_MODEL_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # 根据显存调整
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch"
        )

        # 定义 data_collator 函数
        def data_collator(data):
            input_ids = []
            attention_mask = []
            labels = []
            for sample in data:
                # 这里假设 processed_data 是一个列表，每个元素对应一个样本的处理结果
                index = dataset.index(sample)  # 获取当前样本在数据集中的索引
                input_ids.append(processed_data[index]['input_ids'])
                attention_mask.append(processed_data[index]['attention_mask'])
                labels.append(processed_data[index]['input_ids'])
            # 将列表转换为张量
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            labels = torch.tensor(labels)
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
        return {"message": "训练完成，LoRA 权重已保存。", "status": 200}
    except Exception as e:
        return {"message": f"训练过程中出现错误: {e}", "status": 500}
