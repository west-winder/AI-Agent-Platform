"""仅定义模型交互协议异常；外部服务异常沿用已有契约。"""


class LLMProtocolError(Exception):
    """模型消息缺少必要字段、响应不一致或不能构成有效回答。"""


class LLMResponseTruncatedError(LLMProtocolError):
    """模型因长度限制截断；V1 不自动续写，也不执行截断的工具调用。"""
