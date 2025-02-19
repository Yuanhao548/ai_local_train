from fastapi import Request, APIRouter
import datetime

from utils.conf import split_resp_think_text
from utils.operate_model import call_lora_model

# 创建一个 APIRouter 实例
router = APIRouter()


# 处理POST请求的端点
@router.post("/")
async def create_item(request: Request):
    try:
        json_post = await request.json()  # 获取 POST 请求的 JSON 数据
        prompt = json_post.get('prompt')  # 获取请求中的提示

        messages = [
            {"role": "user", "content": prompt}
        ]

        # 调lora模型进行对话生成
        response = call_lora_model(messages)
        print("response: ", response)

        think_content, answer_content = split_resp_think_text(response)  # 调用split_text函数，分割思考过程和回答
        now = datetime.datetime.now()  # 获取当前时间
        time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
        # 构建响应JSON
        answer = {
            "response": response,
            "think": think_content,
            "answer": answer_content,
            "status": 200,
            "time": time
        }
        # 构建日志信息
        log = f"[{time}], prompt:\"{prompt}\", response:\"{repr(response)}\", think:\"{think_content}\", answer:\"{answer_content}\""
        print(log)  # 打印日志
        return answer  # 返回响应
    except Exception as e:
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        error_log = f"[{time}] Error: {str(e)}"
        return {"status": 500, "message": f"{error_log}"}
