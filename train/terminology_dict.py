# 在工程初始化时加载术语词典，并将其存储为内存中的字典或列表，方便快速查询。
import json


class TerminologyDictionary:
    """
    专业领域数据词典结构
    {
        "terms": [
            {
                "term": "等价类划分",
                "definition": "将输入域划分为若干等价类，每个等价类中的输入具有相同的测试效果",
                "category": "测试方法",
                "synonyms": ["等价类测试", "等价类分析法"]
            },
            {
                "term": "边界值分析",
                "definition": "测试输入域的边界值，通常包括最小值、最大值和临界值",
                "category": "测试方法",
                "synonyms": ["边界测试", "边界值法"]
            },
            {
                "term": "并发测试",
                "definition": "模拟多个用户同时操作，验证系统在高并发情况下的性能",
                "category": "性能测试",
                "synonyms": ["压力测试", "负载测试"]
            }
        ]
    }
    """

    def __init__(self, dictionary_path):
        with open(dictionary_path, "r", encoding="utf-8") as f:
            self.terms = json.load(f)["terms"]

    def get_definition(self, term):
        """获取术语的定义"""
        for item in self.terms:
            if item["term"] == term or term in item.get("synonyms", []):
                return item["definition"]
        return None

    def get_synonyms(self, term):
        """获取术语的同义词"""
        for item in self.terms:
            if item["term"] == term:
                return item.get("synonyms", [])
        return []

    def get_all_terms(self):
        """获取所有术语"""
        return [item["term"] for item in self.terms]


# 加载术语词典
terminology_dict = TerminologyDictionary("terminology_dictionary.json")