# 代码阅读导读

建议打开本目录，按入口 → MCP 适配 → 业务抓取 → 安全/网络 → HTML → 测试的顺序阅读。文件链接均相对本文目录。

## 1. 从入口看到 stdio

[__main__.py](web_page_fetch/__main__.py) 是 `python -m web_page_fetch` 的入口：先配置本进程安全日志，创建 Server 后再复核 handler，执行 `server.run(transport="stdio")`。

stdio 没有 HTTP 端口。Host 中的 MCP Client 启动该 Python 子进程，以 stdin 写入协议消息，以 stdout 读取响应。SDK 负责 JSON-RPC、initialize 握手、请求派发和输入 EOF 处理。stderr 是受控日志。

## 2. 工具如何注册及返回

[server.py](web_page_fetch/server.py) 的 `create_server` 创建官方 `FastMCP("Web Page Fetch MCP Server")`。

其中的 `@mcp.tool(structured_output=True)` 注册唯一异步工具 fetch_web_page。参数 url 的 str 注解产生必填字符串 Schema；返回类型 [FetchWebPageResult](web_page_fetch/contracts.py) 产生五字段 outputSchema。返回模型实例后，SDK 校验并构造 structuredContent 及兼容 TextContent，代码不直接拼装 JSON-RPC。

这里的 Fetcher 注入用于复用相同 MCP 工具适配：生产绑定 WebPageFetcher，测试子进程绑定 FakeFetcher。生产入口没有放宽 SSRF 的开关。

## 3. 一次请求如何运行

[fetcher.py](web_page_fetch/fetcher.py) 的 `WebPageFetcher.fetch` 建立 30 秒绝对截止时间，再调用 `_fetch`。重定向循环共享这个截止时间，每跳创建和关闭独立 Client。

先调用 validate_url，再 resolve_public_target。拿到全部通过检查的 IP 后才创建 PinnedIPTransport。Fetcher 通过 client.stream 发起 GET，只接受安全状态、媒体类型和编码。

响应先检查 Content-Encoding，仅 identity 或缺失可接受；随后只接受 text/html，再检查 Content-Length，最后通过 aiter_raw 按实际字节数限制 2 MiB。没有通过自动解压后的文本计算原始字节大小。

重定向 Location 以逻辑 URL 为基准；HTTPS 到 HTTP 在下一请求前拒绝。下一跳重新进行 URL、DNS 和全部 IP 检查，不继承 Cookie 或连接。

## 4. SSRF 在哪里执行

[security.py](web_page_fetch/security.py) 分三个步骤：

1. validate_url：限制协议及端口，拒绝凭据/控制字符，规范化 IDNA 与单个尾点，统一主机身份。
2. validate_public_ip：拒绝非公网地址及已知特殊平台/转换地址，IPv4-mapped IPv6 全部拒绝。
3. resolve_public_target：IP 字面量直接检查；域名异步解析，所有候选均通过后才选第一个。混合公网/私网整体拒绝。

`ValidatedTarget` 将逻辑 URL、统一 host、port 和候选 IP 保存在一起。它不缓存全局 DNS 结果。

[PinnedIPTransport](web_page_fetch/fetcher.py) 在 HTTPX Transport 边界把连接 URL 的 host 替换为选定数字 IP，Host 和 sni_hostname 保留规范化原域名。它委托真实 AsyncHTTPTransport，后者进入 HTTPcore。

原域名不再用于 TCP 解析。HTTPcore 1.0.9 的 SNI 扩展直接传给 TLS，必须是 str。HTTPS 验证保持开启；没有 verify=False 或代理兜底。

## 5. HTML 如何解码和解析

[html_text.py](web_page_fetch/html_text.py) 先由 decode_html 选择 BOM → HTTP charset → 前 1,024 字节的 meta charset → UTF-8。无效标签跳过；已选编码的坏字节替换，不进行猜测。

ReadableHTMLParser 维护标签栈、隐藏状态及 article/main/body/fallback 文本区域。脚本、样式、模板、隐藏内容和 head 不进入正文。result 按优先级选第一个清理后非空区域，保留段落，截断标题和正文。truncated 只反映正文超限。

extract_html_text 分块 feed，在块间及完成后检查预算。HTMLParser 是同步解析器，不能把这个检查解释为进程级硬超时。

## 6. 错误如何返回 Client

[errors.py](web_page_fetch/errors.py) 仅允许固定 ErrorCode 映射到固定消息。网络和 DNS 异常在边界转换，原始异常字符串不进入公开消息。

server 的工具捕获 FetchError 后只保留代码；其他普通异常映射 INTERNAL_ERROR。它退出 except 块后再创建 ToolError，避免原异常作为该错误的隐式上下文。SDK 仍会包装自己的安全 ToolError，并返回 isError=true。取消是 BaseException 路径，不被 except Exception 吞掉。

configure_safe_logging 在 SDK 创建前后建立/复核本进程 handler，并将 Python warnings 接入 logging；_SafeHandler 只输出固定事件和枚举代码，不格式化第三方异常、warning 原文或 traceback。它不会改变主项目日志配置，因为 Server 是独立子进程。

SDK 参数校验及畸形协议消息有单独的错误边界，详见 README；不要把应用错误转换理解为 SDK 所有错误都已经统一。

## 7. 资源怎样关闭

Fetcher 的 Client 和流式 Response 都使用 async with，正常、重定向、HTTP 错误、超限、超时或取消都会退出上下文。PinnedIPTransport.aclose 委托关闭实际 Transport。

DNS 的线程式系统解析可能不能在取消时立即终止，这是已记录限制。Server 收到 stdin EOF 由官方 SDK 结束；Client 使用官方 stdio_client 的上下文管理资源。

## 8. 用测试对照实现

- [test_contracts.py](tests/test_contracts.py)：严格输出及固定错误。
- [test_security.py](tests/test_security.py)：URL、尾点、IDNA、地址族、DNS 混合结果/失败。
- [test_connection_security.py](tests/test_connection_security.py)：真实 Transport 链，记录 connect_tcp、线上 Host 字节和 TLS 验证参数；不是实际握手。
- [test_fetcher.py](tests/test_fetcher.py)：流式原始字节、头部、重定向/降级、超时、取消和关闭。
- [test_html_text.py](tests/test_html_text.py)：编码冲突、区域优先级、非正文清理及长度边界。
- [test_stdio.py](tests/test_stdio.py)：官方 Client 与真实子进程，成功/失败/Schema/恢复/正常退出/日志 canary。
- [test_logging.py](tests/test_logging.py)：启动及运行时 Python warning 不得绕过 stderr 白名单。

Phase 3 的 [acceptance_client.py](tests/support/acceptance_client.py) 启动未替换组件的生产 Server 执行真实公网抓取；它不是 pytest 用例，不让确定性回归依赖互联网。结果见 [phase3-acceptance.json](phase3-acceptance.json)。

stdio 测试仅观察 SDK 的进程创建及强制终止函数，没有替换其协议通信。测试专用 [fixture_server.py](tests/support/fixture_server.py) 注入 FakeFetcher，不执行外部请求；生产入口测试另外验证真实安全拒绝路径。
