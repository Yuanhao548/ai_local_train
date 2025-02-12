import os

from modelscope import snapshot_download

from pathlib import Path

# 获取当前文件的路径对象
current_file_path = Path(__file__)
# 获取当前文件的父目录
root_path = current_file_path.parent.parent
model_dir = Path(os.path.join(root_path, 'deepseek_R1_Distill_Qwen_7B'))


# 判断目录是否存在
def model_snapshot_download() -> None:
    global model_dir
    if not model_dir.exists():
        # 如果不存在，则创建目录
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"目录 {model_dir} 已创建。")
        model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', cache_dir=model_dir, revision='master')
    else:
        print(f"目录 {model_dir} 已存在。")
