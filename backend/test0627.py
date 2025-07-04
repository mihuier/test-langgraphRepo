


# #################--------------###################
import datetime
import json
from typing import Any, Dict, List, cast
from langchain_core.messages import AIMessage
from langchain.chat_models import ChatOpenAI

# Qwen模型配置
BASIC_API_KEY = "sk-0ec88898edb24291a7268f6684868552"  # mihuier-new
BASIC_MODEL = "qwen-plus"
BASIC_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

class State:
    """会话状态类，用于存储对话历史"""
    def __init__(self, messages: List[Dict[str, str]] = None, is_last_step: bool = False):
        self.messages = messages or []
        self.is_last_step = is_last_step

class Configuration:
    """配置类，用于存储系统提示信息"""
    @staticmethod
    def from_context():
        return Configuration()
    
    def __init__(self):
        self.model = BASIC_MODEL
        self.system_prompt = "You are a helpful assistant. Current time: {system_time}"

def object_to_dict(obj: Any) -> Any:
    """
    递归将对象转换为可序列化的字典结构
    
    Args:
        obj: 要转换的对象
    Returns:
        可序列化的字典/列表/基本类型
    """
    # 处理基本类型
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    # 处理列表和元组
    if isinstance(obj, (list, tuple)):
        return [object_to_dict(item) for item in obj]
    
    # 处理字典
    if isinstance(obj, dict):
        return {key: object_to_dict(value) for key, value in obj.items()}
    
    # 处理自定义对象
    result = {}
    for attr in dir(obj):
        if attr.startswith("__"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception as e:
            result[attr] = f"Error accessing attribute: {str(e)}"
            continue
        
        # 排除方法和函数（只保留属性值）
        if callable(value):
            result[attr] = f"Callable: {value.__name__}"
        else:
            result[attr] = object_to_dict(value)
    return {obj.__class__.__name__: result}

async def call_model_and_save() -> None:
    """调用Qwen模型并将返回的消息属性保存为JSON文件"""
    # 初始化配置和状态
    configuration = Configuration.from_context()
    state = State(messages=[{"role": "user", "content": "7月1日是什么节日?"}])
    
    # 初始化模型
    model = ChatOpenAI(
        model_name=BASIC_MODEL,
        api_key=BASIC_API_KEY,
        base_url=BASIC_BASE_URL,
        temperature=0.0,
    )
    
    # 格式化系统提示
    system_message = configuration.system_prompt.format(
        system_time=datetime.datetime.now().isoformat()
    )
    
    print("Sending request to Qwen model...")
    
    # 获取模型响应
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )
    
    # 转换响应为字典并保存为JSON
    response_dict = object_to_dict(response)
    output_filename = "qwen_response_messages.json"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(response_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Message attributes saved to {output_filename}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(call_model_and_save())