import json

from peft import PeftModel

from trained.terminology import terminology_dict
from trained.train import train_tokenizer, train_model
from utils.conf import cached_tokenize, generate_ai_response
from utils.constant import VALID_DATA_SET_PATH, DEVICE, TRAINED_LORA_WEIGHTS_MODEL_DIR


# ---------- 验证模块 ----------
class TestCaseValidator:
    """测试用例验证器"""

    def __init__(self, base_model_tokenizer, base_model, lora_model_path, terminology):
        self.tokenizer = base_model_tokenizer
        self.model = PeftModel.from_pretrained(base_model, lora_model_path)     # 将lora权重模型加进基础模型
        self.model.to(DEVICE)
        self.terminology_dict = terminology
        self.tokenizer_pad_token_id = self.get_pad_token_id()

    def postprocess_output(self, text):
        """对生成的文本进行术语标准化"""
        for term in self.terminology_dict.get_all_terms():
            if term in text:
                text = text.replace(term, f"[{term}]")  # 标记术语
        return text

    def generate_test_points(self, requirement):
        """生成测试点"""
        prompt = f"""根据以下需求分析测试点：
需求：{requirement}
测试点分析："""

        inputs = cached_tokenize(self.tokenizer, prompt, DEVICE)
        raw_output = generate_ai_response(inputs, self.model, self.tokenizer_pad_token_id, self.tokenizer)
        print("生成的测试点是：", raw_output)

        # 提取测试点部分
        test_points = raw_output.split("生成的测试用例：")[0].strip()
        return self.postprocess_output(test_points)

    def get_pad_token_id(self):
        # 检查并设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用eos_token作为pad_token

        # 确认pad_token_id
        pad_token_id = self.tokenizer.pad_token_id
        return pad_token_id

    def get_input_text(self, messages):
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return input_text

    def generate_testcase(self, messages):
        """生成测试用例，并进行术语标准化"""
        requirement = self.get_input_text(messages)

        # 先生成测试点
        test_points = self.generate_test_points(requirement)

        prompt = f"""根据以下需求生成测试用例：
需求：{requirement}
测试点：{test_points}
测试点分析："""

        inputs = cached_tokenize(self.tokenizer, prompt, DEVICE)
        raw_output = generate_ai_response(inputs, self.model, self.tokenizer_pad_token_id, self.tokenizer)
        print("生成的测试用例是：", raw_output)
        test_cases = raw_output.split("生成的测试用例：")[1].strip()
        return test_points, self.postprocess_output(test_cases)


# 初始化加载了lora权重的模型
validator = TestCaseValidator(train_tokenizer, train_model, TRAINED_LORA_WEIGHTS_MODEL_DIR, terminology_dict)


def load_validation_data(validation_path):
    """加载验证集"""
    with open(validation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_on_validation_set(validator, validation_data):
    """在验证集上评估模型"""
    results = []
    for item in validation_data:
        requirement = item["requirement"]
        expected_points = item["test_points"]

        # 生成测试点和测试用例
        generated_points, generated_cases = validator.generate_testcase(requirement)

        # 计算覆盖率
        coverage = calculate_coverage(generated_points, expected_points)

        # 保存结果
        results.append({
            "requirement": requirement,
            "generated_points": generated_points,
            "generated_cases": generated_cases,
            "expected_points": expected_points,
            "coverage": coverage
        })
    return results


def calculate_coverage(generated_points, expected_points):
    """计算测试点覆盖率"""
    generated_set = set(generated_points)
    expected_set = set(expected_points)
    return len(generated_set & expected_set) / len(expected_set)


def calculate_metrics(results):
    """计算整体评估指标"""
    total_coverage = sum(item["coverage"] for item in results) / len(results)
    return {
        "average_coverage": total_coverage
    }


def save_evaluation_results(results, output_path):
    """追加保存评估结果"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def start_valid():
    # 加载验证集
    validation_data = load_validation_data(VALID_DATA_SET_PATH)

    # 在验证集上评估模型
    results = evaluate_on_validation_set(validator, validation_data)

    # 计算评估指标
    metrics = calculate_metrics(results)
    print(f"平均覆盖率：{metrics['average_coverage'] * 100:.1f}%")

    # 保存评估结果
    save_evaluation_results(results, "evaluation_results.json")


# 示例
if __name__ == '__main__':
    start_valid()
