from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# 加载领域文本（如金融协议、医疗标准）
domain_corpus = load_dataset("text", data_files="domain_specific_docs.txt")

# 继续预训练
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)


def tokenize_domain_text(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def save_fine_model(trainer):
    # 保存微调后的模型和分词器
    trainer.save_model("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")


class TestCaseTrain:
    def __init__(self):
        pass

    @staticmethod
    def domain_specific_train():
        domain_dataset = domain_corpus.map(tokenize_domain_text, batched=True)
        training_args = TrainingArguments(
            output_dir="./domain_pretrain",
            per_device_train_batch_size=4,
            num_train_epochs=1,
            learning_rate=5e-5,
            fp16=True
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=domain_dataset
        )
        trainer.train()

    @staticmethod
    def test_point_train():
        # 输入格式：[需求]... [领域]... [关键词]...
        # 输出格式：[测试点]... [术语表]...
        training_args = TrainingArguments(
            output_dir="./test_point_model",
            per_device_train_batch_size=4,
            num_train_epochs=3,
            learning_rate=3e-5,
            logging_steps=100,
            evaluation_strategy="steps"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=load_dataset("json", data_files="processed_data.jsonl")["trained"],
            eval_dataset=load_dataset("json", data_files="val_data.jsonl")["validation"]
        )
        trainer.train()

    @staticmethod
    def generate_test_suit():
        # 输入格式：[需求]... [测试点]... [领域术语]...
        # 输出格式：[测试用例]...
        training_args = TrainingArguments(
            output_dir="./test_case_model",
            per_device_train_batch_size=4,
            num_train_epochs=3,
            learning_rate=2e-5
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=load_dataset("json", data_files="test_case_data.jsonl")["trained"]
        )
        trainer.train()