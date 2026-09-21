class ExternalServiceError(Exception):
    """
    外部服务调用失败的基础异常。
    """


class ExternalTimeoutError(ExternalServiceError):
    """
    外部服务调用超时。
    """


class ExternalRequestError(ExternalServiceError):
    """
    外部服务请求或配置存在问题。
    """