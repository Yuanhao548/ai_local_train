from fastapi import Request, APIRouter
import datetime
import re

from utils.operate_model import call_deepseek_model

# 创建一个 APIRouter 实例
router = APIRouter()


# 设置设备参数
# DEVICE = "cuda"  # 使用CUDA
# DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空


# 文本分割函数
def split_text(text):
    pattern = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL)  # 定义正则表达式模式
    match = pattern.search(text)  # 匹配 <think>思考过程</think>回答

    if match:  # 如果匹配到思考过程
        think_content = match.group(1).strip()  # 获取思考过程
        answer_content = match.group(2).strip()  # 获取回答
    else:
        think_content = ""  # 如果没有匹配到思考过程，则设置为空字符串
        answer_content = text.strip()  # 直接返回回答

    return think_content, answer_content


# 处理POST请求的端点
@router.post("/")
async def create_item(request: Request):
    try:
        json_post = await request.json()  # 获取 POST 请求的 JSON 数据
        prompt = json_post.get('prompt')  # 获取请求中的提示

        messages = [
            {"role": "user", "content": prompt}
        ]

        # 调用模型进行对话生成
        response = call_deepseek_model(messages)

        think_content, answer_content = split_text(response)  # 调用split_text函数，分割思考过程和回答
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
