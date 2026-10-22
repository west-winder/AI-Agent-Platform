
class ToolRegistryError(Exception):
    """工具注册表相关错误的基类。"""


class DuplicateToolError(ToolRegistryError):
    """工具名称重复注册。"""


class ToolNotFoundError(ToolRegistryError):
    """请求的工具不存在。"""


class ToolExecutionError(Exception):
    """工具执行过程中的已知业务错误。"""

    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message
        self.retryable = retryable
