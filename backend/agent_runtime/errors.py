"""模型交互和上下文构造异常；外部服务异常沿用已有契约。"""


class LLMProtocolError(Exception):
    """模型消息缺少必要字段、响应不一致或不能构成有效回答。"""


class LLMResponseTruncatedError(LLMProtocolError):
    """模型因长度限制截断；V1 不自动续写，也不执行截断的工具调用。"""


class ContextInputError(LLMProtocolError, ValueError):
    """History 或当前请求起点无效；错误信息不包含用户内容。"""


class ContextBudgetExceededError(Exception):
    """必要上下文超过本地预算；不得作为工具错误反馈或自动重试。"""
