# M7.2 Web Page Fetch MCP Server — 实施基线

- 日期：2026-11-02（Asia/Shanghai）
- 状态：Phase 1 修订已纳入；Phase 2 实现及 Phase 3 独立验收已完成，未接入平台。实际证据见 README 的 Phase 3 章节。
- 本文记录正式设计决策；实际实现、测试结果和证据边界见 README.md、pytest-results.xml。
- Phase 1 修订轮仅新增本文；Phase 2 的实现和依赖安装限定在本独立目录，不修改主项目配置。

## 1. 范围与固定决策

独立 MCP Server，仅通过 stdio 提供 `fetch_web_page`。不接入 AI Agent Platform，不修改 Agent Runtime、Tool Registry、Tool Dispatcher、Weather Tool 或主项目依赖。不实现 HTTP Transport、JS 渲染、登录、Cookie 管理、搜索、批量抓取或资源下载。不执行 Commit、Push 或 Tag。

固定 Python 3.12（当前已核验版本 3.12.9）、官方 MCP Python SDK 1.26.0、FastMCP、HTTPX 0.28.1、HTTPcore 1.0.9、标准库 HTMLParser。仅允许 HTTP 80、HTTPS 443。

本次修订覆盖此前方案中相冲突的条目：

1. 编码优先级改为 BOM → HTTP charset → HTML meta charset → UTF-8。
2. 第一版仅接受 `text/html`，拒绝 `application/xhtml+xml`。
3. 域名单个 trailing dot 去除后统一用于 DNS、Host、SNI。
4. 连接安全证据必须经过真实 HTTPX/HTTPcore Transport 调用链。
5. 先检查 Content-Encoding，再用 `aiter_raw()` 统计原始响应体字节。
6. 拒绝 HTTPS → HTTP 降级重定向。
7. 日志脱敏覆盖应用、异常链及 SDK/HTTP 客户端日志路径。

其余已评审的 Phase 1 决策保持不变。

## 2. 目录与职责

根目录：`F:\AI_Agent_Platform\mcp_servers\web_page_fetch\`。

```text
web_page_fetch/
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── web_page_fetch/
│   ├── __init__.py
│   ├── __main__.py
│   ├── contracts.py
│   ├── errors.py
│   ├── server.py
│   ├── security.py
│   ├── fetcher.py
│   └── html_text.py
└── tests/
    ├── test_security.py
    ├── test_fetcher.py
    ├── test_connection_security.py
    ├── test_html_text.py
    ├── test_stdio.py
    └── support/
        ├── __init__.py
        └── fixture_server.py
```

目录中的实现及测试文件已在 Phase 2 创建；另有 READING_GUIDE.md、DEPENDENCIES.json 和 pytest-results.xml 用于导读与交付证据。

| 文件 | 职责与核心接口 |
| --- | --- |
| contracts.py | `FetchWebPageResult`：Pydantic 五字段成功输出 |
| errors.py | `FetchError`：固定错误代码和固定安全消息 |
| security.py | `validate_url`、`validate_public_ip`、`resolve_public_target`；`ValidatedTarget` 保存逻辑 URL、规范化主机、端口及已检查 IP |
| fetcher.py | `WebPageFetcher.fetch` 管理预算、重定向和响应；`PinnedIPTransport` 固定连接 IP，委托标准异步 Transport |
| html_text.py | 编码选择、`ReadableHTMLParser`、`extract_html_text` |
| server.py | `create_server(fetcher)`，注册唯一工具，安全错误转换及独立进程日志策略 |
| __main__.py | 绑定生产依赖，启动 stdio，无测试模式开关 |
| tests/support/fixture_server.py | 测试专用子进程入口，通过 create_server 注入 Fake Fetcher |
| README.md | 安装、运行、契约、实际测试证据及已知限制 |

仅在 Fetcher、DNS Resolver、底层 HTTP Transport 边界进行测试注入。不增加生产环境放宽 SSRF 的配置。

## 3. 独立环境、依赖与启动

requirements.txt 的实施内容：

```text
mcp==1.26.0
httpx==0.28.1
httpcore==1.0.9
pydantic==2.13.4
```

requirements-dev.txt 的实施内容：

```text
-r requirements.txt
pytest==9.1.1
```

直接依赖及关键传输依赖固定；这不等于完整传递依赖锁。安装后运行 pip check，并记录实际依赖版本。不增加 mcp[cli] 或额外 HTML/编码检测依赖；异步测试使用 asyncio.run()。

以下是 Phase 2 环境及启动命令；本机实际执行与临时目录适配记录见 README：

```powershell
Set-Location 'F:\AI_Agent_Platform\mcp_servers\web_page_fetch'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B -m web_page_fetch
```

测试命令：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests -q
```

仓库现有 `.gitignore` 的 `.venv/` 已通过 git check-ignore 核验覆盖此处嵌套环境。无需修改主项目配置。Client 使用独立解释器绝对路径，cwd 设置为此独立目录。不引入打包安装配置。

## 4. MCP 契约与生命周期

输入仅有必填的 `url: string`。成功输出使用 Pydantic `FetchWebPageResult`，五字段全部必有、非 null，禁止额外字段：

| 字段 | 类型 | 约定 |
| --- | --- | --- |
| title | string | 清理后最多 500 字符，无标题时为空 |
| url | string | 最终成功的规范化逻辑 URL，移除 fragment，不替换为连接 IP |
| content | string | 清理后最多 12,000 字符，无正文时为空 |
| truncated | boolean | 仅表示清理后正文超过上限而被截断 |
| content_type | string | 固定为规范化媒体类型 text/html，不含 charset |

使用 `@mcp.tool(structured_output=True)` 和模型返回类型，由 SDK 生成 inputSchema/outputSchema、校验并生成 structuredContent 及兼容文本内容；不增加 result 包装，不手写 JSON-RPC。

固定应用错误分类：INVALID_URL、TARGET_NOT_ALLOWED、DNS_ERROR、TIMEOUT、NETWORK_ERROR、HTTP_ERROR、REDIRECT_ERROR、UNSUPPORTED_CONTENT_TYPE、UNSUPPORTED_CONTENT_ENCODING、RESPONSE_TOO_LARGE、INVALID_RESPONSE、INTERNAL_ERROR。

工具边界转换为只含固定代码及固定消息的 ToolError，SDK 返回 isError=true，可能附加固定工具执行上下文。代码不是 JSON-RPC error code。未知执行异常转换为 INTERNAL_ERROR；取消信号继续传播。SDK 在工具执行前生成的缺参/类型校验错误仍属于 SDK 错误，不承诺使用应用错误代码。

启动使用 `mcp.run(transport="stdio")`。stdin/stdout 仅传协议消息，stderr 输出安全日志。初始化、编解码、EOF 关闭和 Client 子进程管理使用官方 SDK。不注册远程日志级别切换或额外资源、提示词。

## 5. 数据流和目标规范化

```text
官方 Client → stdio → SDK 校验 → fetch_web_page
  → 30 秒共享预算
  → URL 校验与规范化
  → DNS 解析（IP 字面量直接检查）
  → 检查全部候选 IP
  → 选择并固定 TCP 目标 IP
  → 构建逻辑 Host / TLS SNI → GET
  → 重定向：关闭响应，检查降级并重新执行完整安全流程
  → 最终响应：状态、Content-Type、Content-Encoding 检查
  → Content-Length 提前检查 + aiter_raw() 实际计数
  → BOM / HTTP charset / meta charset / UTF-8 解码
  → HTMLParser → 正文选择与清理 → 截断
  → Pydantic → SDK structuredContent → Client
```

逻辑 URL 与连接 URL 分离。前者用于重定向、返回结果和身份；后者仅将主机替换为检查后的数字 IP。

URL 规则：仅 HTTP:80 / HTTPS:443，默认端口允许省略；拒绝凭据、空主机、控制字符、反斜杠、非法端口。移除 fragment，保留 path/query。拒绝 localhost 及其子域形式。

域名处理顺序：提取域名 → IDNA ASCII、小写 → 去除单个末尾点 → 校验标签与 localhost 规则 → 构建唯一规范化逻辑 URL。IDNA 映射后的分隔符也必须经过相同校验。多个连续末尾点、空标签和空域名拒绝，不用 rstrip 静默接受多个尾点。IP 字面量独立处理，不按域名去尾点。

DNS、HTTP Host 和 TLS server_hostname 必须来自同一规范化主机字段，不分别重新派生。example.com 与 example.com. 使用同一无尾点身份。覆盖大小写、IDNA、带尾点 localhost、重定向带尾点域名等测试。

## 6. IP、DNS 与连接绑定

- IPv4：要求公网单播，结合 is_global、地址类别及明确特殊范围拒绝策略；不能只用 is_private。
- IPv6：仅允许经检查的公网单播，拒绝 zone/scope ID、IPv4-mapped IPv6（包括映射公网 IPv4），保守拒绝 NAT64、6to4、Teredo 等转换/隧道范围。
- 拒绝私网、回环、链路本地、共享地址、未指定、组播、保留及其他非公网目标。
- 元数据/平台服务防护覆盖 169.254.169.254、100.100.100.200、168.63.129.16 等，IPv6 ULA 同样拒绝；不能仅使用元数据域名黑名单。
- DNS 全部候选 IP 先检查再选择。混合公网与非公网时整个目标拒绝；公网结果去重后按解析器顺序选第一个，不自动切换 IP 重试。
- DNS 空结果、异常及超时均安全失败，不发 HTTP 请求。检查行为以当前 Python 3.12.9 为准。

HTTPX/HTTPcore 连接约束：

- 使用数字 IP 构建连接 URL，域名不再交给 HTTP 客户端重新解析。
- 域名请求 Host 和 HTTPS sni_hostname 均为同一个无尾点 IDNA ASCII 主机。
- HTTPcore 1.0.9 将 sni_hostname 直接传给 backend.start_tls；使用 Python str，不传 bytes，不依赖其他版本的自动解码行为。
- HTTPS 保持 verify=True，SSLContext.check_hostname=true、verify_mode=CERT_REQUIRED。直接 IP URL 按原 IP 验证 IP SAN，不借用域名。
- trust_env=False；显式委托的 AsyncHTTPTransport 同样不信任环境配置；无代理、自动重试或自动重定向。
- 使用 HTTP/1.1，每跳独立 Client/Transport，响应和连接均关闭，不跨跳共享连接或 Cookie。

安全论断限定为：绑定已检查 IP，避免原域名检查后被 HTTP 客户端再次解析的重绑定路径。不宣称完全解决所有 SSRF，不控制透明代理、特殊路由或网络自定义转换。

## 7. 重定向、预算与原始响应体

仅跟随 301/302/303/307/308，始终 GET。最多跟随三次，即最多四次请求。相对 Location 根据逻辑 URL 解析；缺失/非法 Location、次数超限均 REDIRECT_ERROR。其他非成功状态安全失败。

HTTPS → HTTP 无条件拒绝，且在向降级目标发送请求前失败；HTTP → HTTPS 允许。同协议跳转也完整重查协议、端口、规范化域名、DNS 和全部 IP。关闭上一响应后再继续，不读取重定向正文，不继承敏感请求头。

| 限制 | 实施规则 |
| --- | --- |
| 总预算 | 单调时钟 + asyncio.timeout_at，共享 30 秒；重定向不重置 |
| DNS | 最多 5 秒，受剩余总预算约束 |
| 连接 | 5 秒 connect timeout |
| 读取 | 10 秒 read timeout，另受总预算限制 |
| 响应体 | 最多 2 MiB = 2,097,152 字节 |
| 标题/正文 | 500 / 12,000 Unicode 字符 |

最终响应先检查头部，且必须在读取或计数字节前检查 Content-Encoding。请求 Accept-Encoding: identity；响应未指定编码或仅为 identity 才可接受，其他编码、组合或歧义编码拒绝。

仅接受媒体类型 text/html（大小写规范化，可带 charset）；缺失、其他类型以及 application/xhtml+xml 均拒绝。先完成头部检查，再用 Content-Length 提前拒绝明显超限，最终以 `response.aiter_raw()` 读取的原始响应体实际字节数为准。这里不计 HTTP framing/TLS 开销。

逐块累计字节数，超过上限立即停止并关闭，不调用 aread、text、content 或 aiter_bytes 获取正文后计算大小。固定流式块大小；Content-Length 缺失或不可信、chunked 响应仍执行同样计数。非法协议长度转换为安全错误。

超限、取消、超时、错误和重定向均保证响应及 Client 关闭。HTML 解析分块投喂，块间及完成后检查截止时间。总预算为协作式应用限制，不是精确进程硬截止。

## 8. HTML 解码与提取

编码优先级严格为：

1. BOM：匹配 UTF-32 必须先于 UTF-16，避免前缀误判；支持相应 UTF BOM，解码不保留 BOM。
2. 有效 HTTP charset。
3. HTML 开头有限窗口内的常见 meta charset / http-equiv charset 声明（拟定前 1,024 字节）。
4. 明确 UTF-8 回退。

未知、无效或不适用于文本解码的 charset 标签跳过，进入下一级；选定有效编码后，不因出现坏字节猜测其他编码，使用 replacement 策略。最后 UTF-8 同样 errors=replace。HTTP charset 与 meta 冲突时 HTTP 优先，BOM 与 HTTP 冲突时 BOM 优先。不添加编码检测依赖，不处理 XML 声明/外部实体。

正文选择：清理后非空 article → main → body；缺少显式 body 时回退到 head 之外文本。同类多个区域按顺序拼接，嵌套不重复，空 article 不阻止回退。

忽略 script/style/noscript/template、注释及明确 hidden / aria-hidden=true 区域。不执行 CSS；不承诺判断外部样式的可见性。第一个 title 用于标题，不进入正文。解码实体、规范化不换行空格、折叠横向空白、保留块级段落边界并合并连续空行。

清理后标题取前 500 字符；正文取前 12,000 字符，不追加省略号。只有正文长度超限才令 truncated=true，恰好等于上限为 false。

## 9. 错误与日志脱敏

生产进程只输出受控事件名、固定错误代码及必要的非敏感计数/耗时。不记录完整 URL、query、主机/IP、请求/响应头、上游正文、原始异常字符串或 repr。结果中的逻辑 URL 是工具契约字段，不应被日志再次记录。

低层异常转换后只向 SDK 暴露固定消息；工具边界避免向外保留可打印的原始 cause，抑制上下文显示。`from None` 只抑制 traceback 展示，并不删除内存中的 __context__，不能据此声称异常链已脱敏。不得使用 logger.exception、exc_info 或异常对象格式化输出原始链。

本轮只读核验发现：MCP SDK 1.26.0 的 Tool.run 会把捕获异常字符串包装到 ToolError，并使用显式 cause；lowlevel Server 存在完整入站消息 DEBUG 日志、流异常 ERROR 日志和 notification 异常 traceback 路径。HTTPX INFO、HTTPcore DEBUG/trace 同样可能带 URL、IP 或异常。

因此生产启动需显式建立仅接受应用受控事件的 stderr 日志出口，屏蔽第三方原始日志记录；不能只依赖默认 INFO 级别，也不能假定 SDK 自动脱敏。配置限于本独立进程，在 SDK 构造前建立出口，构造后复核有效 handler，避免重复 handler 绕过策略。Python warnings 通过 logging.captureWarnings(True) 接入相同出口，避免直接写入 stderr。不允许第三方环境配置重新启用原始日志。此项由 Phase 3 实际启动 warning 暴露并补齐，未改变原有安全目标。

应用执行路径的原始异常、cause/context、SDK 日志和 stderr 使用敏感 canary 验证。测试还检查相关 MCP 日志通知。SDK 工具执行前的参数校验可能回显调用方的非法值；畸形协议消息还可能触发 SDK 自身错误或日志通知。这属于单独的 SDK 协议边界，不能据应用日志过滤声称已全部净化，也不为此自行改写 JSON-RPC。Phase 2 必须记录实际行为；发现本工具正常调用路径泄露内部数据则不通过验收。

## 10. 测试架构及证据边界

| 类别 | 方法 | 证明范围 | 不能证明 |
| --- | --- | --- | --- |
| Security | Fake DNS、URL/IP 参数化 | 尾点/IDNA、地址规则、混合结果、解析失败 | 实际 TCP 绑定 |
| Fetcher | 注入 Resolver、流式 MockTransport | 状态、重定向/降级、预算、响应头、原始流大小、关闭 | 真实连接/TLS |
| 连接安全 | 真实 HTTPX AsyncHTTPTransport → HTTPcore pool/connection → 记录型 backend/stream | connect_tcp 实参、线上的 HTTP Host、TLS 参数、DNS 无二次域名解析 | 真实证书链握手、特殊网络路由 |
| HTML | 固定字节/HTML 样本 | 编码优先级、正文选择、清理、截断 | 浏览器最终呈现 |
| MCP stdio | 官方 Client + 真子进程 | 初始化、Schema、工具发现/调用、structuredContent、isError、关闭 | Fake Fetcher 不证明生产抓取 |

连接安全测试不得以检查 HTTPX Request 替代连接证据。保留真实 HTTPX/HTTPcore Transport 链路，仅在网络 backend/stream 边界注入，不直接伪造 Transport 返回结果。记录：

- 实际传给 connect_tcp 的目标必须为已验证数字 IP，端口为 80/443。
- 从 stream.write 的 HTTP 请求字节验证 Host，而非只查看 Request 对象。
- 从 start_tls 验证规范化 str server_hostname、check_hostname=true 和 CERT_REQUIRED。
- 模拟证书校验失败时请求失败，不回退 verify=False、不接受内容。
- DNS spy 区分地址检查时解析与连接阶段；模拟后续解析返回私网，确认不会再解析原域名。
- FQDN 尾点、IDNA、跨域重定向、相同 IP 不同域名的连接隔离。

记录型 stream 能证明验证配置及失败传播，不能替代实际 TLS 证书校验成功/失败的握手证据；验收不得混淆。若未执行实际 TLS 握手测试，应明确标注。

Fetcher 至少覆盖原始要求的十三项，以及 XHTML 拒绝、压缩响应在读取前拒绝、aiter_raw 超限（含无/不可信 Content-Length）、HTTPS 降级不发送目标请求、3 次边界、混合 DNS、取消关闭和共享总预算。

HTML 测试覆盖 BOM/HTTP/meta 冲突、无效 charset、UTF-8 replacement、UTF-32/UTF-16 BOM 区分、article/main/body 回退及恰好/超过长度边界。

stdio 两组：生产入口验证启动/发现/localhost 拒绝；测试入口通过 create_server 注入 Fake Fetcher，验证确定性成功、固定错误及失败后继续调用。生产无 SSRF 放宽开关、无 tests 导入。

官方 Client 使用 StdioServerParameters、stdio_client、ClientSession，执行 initialize/list_tools/call_tool。捕获 stderr 和工具错误检查 canary。关闭检查 stdin EOF 后正常退出及退出码，不能只看 Client 上下文退出，因为 SDK 有强制终止兜底。

## 11. 实施顺序及已知限制

顺序保持：独立环境 → 契约/错误/Security 及测试 → 固定 IP Transport 与连接测试 → HTML 解码/解析测试 → Fetcher 测试 → FastMCP/模块入口 → 真实 stdio 验收 → README 证据及改动范围复核。每个主要组件完成后报告文件、职责、关键逻辑和测试结果。

已知限制：系统 DNS 等待超时未必终止底层线程；IP 分类随版本变化；固定 IP 不控制透明代理或特殊路由；记录型 TLS 测试不证明真实握手；标准库解析畸形 HTML 有限；拒绝压缩、XHTML、特殊 IPv6 和备用 IP 重试会降低兼容性；总预算受同步处理及清理时间影响。传递依赖未完整锁定。SDK 非法输入/畸形协议消息的错误文本另有边界。

## 12. 只读核验来源

- [MCP Python SDK v1.26.0 README](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/README.md)
- [Tool.run：异常包装与 cause](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/src/mcp/server/fastmcp/tools/base.py)
- [SDK lowlevel Server：参数校验、日志及错误路径](https://github.com/modelcontextprotocol/python-sdk/blob/v1.26.0/src/mcp/server/lowlevel/server.py)
- [HTTPcore 1.0.9：连接 IP 与 sni_hostname 传递](https://github.com/encode/httpcore/blob/1.0.9/httpcore/_async/connection.py)
- [HTTPcore 1.0.9 AnyIO：str server_hostname 和 TLS 包装](https://github.com/encode/httpcore/blob/1.0.9/httpcore/_backends/anyio.py)
- [HTTPX 0.28.1：扩展传递](https://github.com/encode/httpx/blob/0.28.1/httpx/_transports/default.py)

已检查本机相应已安装源码；未通过安装新版本替代核验。本文测试条目为验收要求，实际通过记录及未验证项以 README 和原始测试报告为准。
