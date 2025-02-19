from fastapi import FastAPI
import torch
import uvicorn

from api.api import router
from utils.snapshot_download import model_snapshot_download
# from trained.train import train

# 创建FastAPI应用
app = FastAPI()
# 挂载路由
app.include_router(router)



# 主函数入口
if __name__ == "__main__":
    # 下载模型
    model_snapshot_download()
    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(
        app, host="127.0.0.1", port=6006, workers=1
    )  # 在指定端口和主机上启动应用
    # train()
