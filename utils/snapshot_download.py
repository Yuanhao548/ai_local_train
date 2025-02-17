from modelscope import snapshot_download

from utils.constant import BASE_MODEL_DIR, BASE_MODEL_NAME


# 判断目录是否存在
def model_snapshot_download(model_dir=BASE_MODEL_DIR) -> None:
    if not model_dir.exists():
        # 如果不存在，则创建目录
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"目录 {model_dir} 已创建。")
        model_dir = snapshot_download(f'deepseek-ai/{BASE_MODEL_NAME}', cache_dir=model_dir, revision='master')
    else:
        print(f"目录 {model_dir} 已存在。")
