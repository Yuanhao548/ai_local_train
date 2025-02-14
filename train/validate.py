import json

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from train.terminology import terminology_dict
from utils.constant import VALID_DATA_SET_PATH, DEVICE, TRAINED_LORA_WEIGHTS_MODEL_DIR


# ---------- 验证模块 ----------
class TestCaseValidator:
    """测试用例验证器"""

    def __init__(self, model_path, terminology_dict):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.to(DEVICE)
        self.terminology_dict = terminology_dict

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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=150,  # 控制生成长度
            temperature=0.7,
            do_sample=True
        )
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取测试点部分
        test_points = raw_output.split("生成的测试用例：")[0].strip()
        return self.postprocess_output(test_points)

    def generate_testcase(self, requirement):
        """生成测试用例，并进行术语标准化"""
        # 先生成测试点
        test_points = self.generate_test_points(requirement)

        prompt = f"""根据以下需求生成测试用例：
需求：{requirement}
测试点分析：{test_points}
测试点分析："""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        test_cases = raw_output.split("生成的测试用例：")[1].strip()
        return test_points, self.postprocess_output(test_cases)


# 初始化加载了lora权重的模型
validator = TestCaseValidator(TRAINED_LORA_WEIGHTS_MODEL_DIR, terminology_dict)


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
