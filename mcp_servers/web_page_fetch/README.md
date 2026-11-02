# Web Page Fetch MCP Server

独立学习实验：官方 MCP Python SDK 1.26.0 + FastMCP + stdio，仅提供 `fetch_web_page`。不依赖或接入 AI Agent Platform，不导入 backend，不读取主项目 .env。

Phase 2 实现及 Phase 3 独立验收均已完成；未接入平台。正式设计见 [ARCHITECTURE.md](ARCHITECTURE.md)，阅读顺序见 [READING_GUIDE.md](READING_GUIDE.md)。Phase 3 结果及证据见下文。

## 安装与运行

PowerShell，Python 3.12：

```powershell
Set-Location 'F:\AI_Agent_Platform\mcp_servers\web_page_fetch'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B -m web_page_fetch
```

最后一行启动后等待 stdin 的 MCP 消息，没有 HTTP 监听端口。通常由官方 MCP Client 作为子进程启动，不在终端输入普通文本。解释器无需激活，Client 的 command 使用本目录 `.venv\Scripts\python.exe` 的绝对路径，args 使用 `-B -m web_page_fetch`，cwd 使用本目录。

生产只需 requirements.txt；requirements-dev.txt 额外安装 pytest。现有仓库 .gitignore 已覆盖嵌套 .venv，不应提交虚拟环境。

本次环境遇到默认 TEMP 权限不足，实际使用以下方式在独立目录内完成 pip 引导；一般环境不必执行：

```powershell
New-Item -ItemType Directory -Force -Path .venv\tmp | Out-Null
$env:TEMP = (Resolve-Path .venv\tmp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-dev.txt
```

## 实际依赖

2026-11-02，Python 3.12.9，独立 .venv：

| 包 | 安装版本 |
| --- | --- |
| mcp | 1.26.0 |
| httpx | 0.28.1 |
| httpcore | 1.0.9 |
| pydantic | 2.13.4 |
| pytest | 9.1.1 |
| anyio | 4.16.0 |
| h11 | 0.16.0 |
| certifi | 2026.10.23 |

全部安装包及版本保存在 [DEPENDENCIES.json](DEPENDENCIES.json)。这是实际环境快照，不是完整锁文件；requirements 固定直接依赖和关键传输依赖。`pip check` 实际结果：`No broken requirements found.`

## 完整目录（省略被 Git 忽略的 .venv）

```text
web_page_fetch/
├── ARCHITECTURE.md
├── README.md
├── READING_GUIDE.md
├── DEPENDENCIES.json
├── pytest-results.xml
├── phase3-tests.xml
├── phase3-acceptance.json
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
    ├── test_contracts.py
    ├── test_security.py
    ├── test_connection_security.py
    ├── test_html_text.py
    ├── test_fetcher.py
    ├── test_stdio.py
    ├── test_logging.py
    └── support/
        ├── __init__.py
        ├── fixture_server.py
        └── acceptance_client.py
```

## 契约与请求流

唯一输入为必填 `url: string`。成功返回 title、url、content、truncated、content_type 五字段。模型在 contracts.py，FastMCP 自动生成 inputSchema/outputSchema 和 structuredContent，不手写 JSON-RPC。

```text
MCP Client → stdio → server.fetch_web_page
 → WebPageFetcher.fetch（共享 30 秒预算）
 → validate_url → resolve_public_target（全部 IP 检查）
 → PinnedIPTransport → HTTPX → HTTPcore → 已验证数字 IP
 → 每跳重新检查，拒绝 HTTPS 降级
 → text/html + identity 编码检查 → aiter_raw（2 MiB 上限）
 → BOM / HTTP / meta / UTF-8 解码
 → article / main / body 提取 → 12,000 字符截断
 → FetchWebPageResult → SDK structuredContent → Client
```

连接/读取超时分别为 5/10 秒，DNS 等待 5 秒，最多跟随三次重定向。仅 HTTP 80 / HTTPS 443。DNS、Host、SNI 使用同一去除单个尾点的 IDNA ASCII 域名；实际连接 URL 使用数字 IP。HTTPcore 1.0.9 的 SNI 扩展必须传 str。

错误使用固定代码和固定消息。工具边界在 except 块之外抛出安全 ToolError，使原始异常不成为该 ToolError 的上下文。SDK 包装后以 isError=true 返回。取消继续传播；流式响应与 Client 在 async with 中关闭。

独立入口配置白名单 stderr handler：只输出 `fetch_failed code=固定枚举值`。不格式化任意第三方 LogRecord，不输出 traceback、完整 URL、query、IP 或原始异常；SDK/HTTPX/HTTPcore 原始日志被过滤。

## Phase 2 实际测试结果

命令（在独立目录内）：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests -q --junitxml=pytest-results.xml -o faulthandler_timeout=15
.\.venv\Scripts\python.exe -m pip check
```

最终实际结果：**142 passed in 5.53s**，无 skipped、failed 或 errors。JUnit 原始报告见 [pytest-results.xml](pytest-results.xml)。此前分批结果为 126 passed；完整首轮为 137 passed，补充边界测试后最终为 142 passed。

| 测试文件 | 通过数 |
| --- | ---: |
| test_contracts.py | 13 |
| test_security.py | 54 |
| test_connection_security.py | 10 |
| test_html_text.py | 21 |
| test_fetcher.py | 42 |
| test_stdio.py | 2 |
| 合计 | 142 |

本机沙箱阻断 Windows asyncio 创建事件循环所需的内部 socket pair，初次异步测试在业务逻辑执行前阻塞，已停止这些进程；最终测试在获得执行权限后完成。测试没有使用外部网站，生产 SSRF 策略未放宽。

首次测试暴露 HTTPX `.host` 为 Unicode 表示的问题：安全身份比较已改用 `.raw_host` 的 ASCII 表示，保留域名规范化及所有安全要求。未删除或降低失败测试。

### 连接层证据

test_connection_security.py 保留真实 `HTTPX AsyncHTTPTransport → HTTPcore AsyncConnectionPool → AsyncHTTPConnection` 链路，仅替换 AnyIO network backend/stream，不使用 MockTransport 证明连接安全：

- 记录 connect_tcp 的 host/port，验证实际传入目标为检查后的数字 IP。
- 从 stream.write 收到的 HTTP/1.1 请求字节中验证 Host。
- 记录 start_tls 的 server_hostname（无尾点 ASCII str）、check_hostname=true、CERT_REQUIRED。
- 覆盖 FQDN、中文 IDNA、原生 IPv6、直接 IPv4/IPv6 URL、HTTP。
- TLS 失败模拟不发送 HTTP 正文，不重试、不关闭证书验证。
- OS getaddrinfo spy 拒绝额外解析，受控 Resolver 记录检查次数。
- 同 IP 跨域跳转分别建立连接和 TLS 身份；同域重定向第二次 DNS 返回私网时，拒绝创建第二条连接。
- 设置本地环境代理后仍直接连接已检查 IP，证明 trust_env 策略。

这些是网络 backend 调用和 TLS 配置证据，不是实际 OS TCP 建连或真实证书链握手证据。记录型 stream 不执行 TLS 握手；Phase 2 未执行真实 TLS 握手测试，Phase 3 另有生产公网请求证据。

### MCP stdio 证据

test_stdio.py 的两组测试均使用官方 SDK ClientSession / stdio_client 和真实 Python 子进程：

1. 生产入口：initialize、唯一工具发现、输入/输出 Schema、localhost/回环/私网拒绝。
2. 测试入口：注入 Fake Fetcher，验证中文结构化成功结果、NETWORK_ERROR、未知异常 INTERNAL_ERROR、失败后再次成功调用。
3. 两组均验证缺参/错误类型的 SDK 工具错误，以及失败后 tools/list 仍可用。
4. 记录进程退出码均为 0；SDK 强制终止函数调用次数为 0，确认 stdin EOF 后正常退出。
5. 捕获 stderr 和 MCP 日志通知：含合成内部地址、query 和原始异常 canary 的第三方 DEBUG/INFO/ERROR/traceback 被过滤，工具错误没有泄露异常链。

成功协议用例的 Fake Fetcher 只证明 MCP 交互，不替代生产 Fetcher 和连接层测试。

## Phase 3 正式验收（2026-11-02）

结论：独立 Server 本轮验收通过。未新增生产功能、放宽安全策略、重构架构或接入平台。未修改主项目依赖。

### 执行命令与完整测试

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests -q --junitxml=phase3-tests.xml -o faulthandler_timeout=15
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B tests\support\acceptance_client.py
```

- 修复前复跑原有测试：142 passed in 6.00s。
- 最终完整回归：**143 passed in 6.60s**，0 failures/errors/skipped。
- pip check：No broken requirements found；没有安装或变更依赖。
- [phase3-tests.xml](phase3-tests.xml)：最终原始 JUnit 报告，包含原 142 项和新增 warnings 脱敏回归。
- [phase3-acceptance.json](phase3-acceptance.json)：最终生产 stdio 公网/异常/退出验收的脱敏记录。
- [acceptance_client.py](tests/support/acceptance_client.py)：可复现的手动验收客户端，不由 pytest 自动执行。

Windows asyncio/stdio 测试及真实网络验收使用获准的沙箱外执行权限。测试未因此修改任何生产规则。

### 公网请求及协议证据

最终记录开始时间为 **2026-11-02 15:13:06 +08:00**。官方 mcp 1.26.0 的 stdio_client / ClientSession 启动独立解释器下的 `python -B -m web_page_fetch`，未替换 Server Fetcher、Resolver 或 Transport。

| 验收项 | 实际结果 |
| --- | --- |
| initialize | Web Page Fetch MCP Server，协商协议 2025-11-25 |
| tools/list | 仅 fetch_web_page，输入 Schema 为必填 url:string |
| 真实目标 | https://example.com/ |
| tools/call | isError=false，首个公网调用耗时 0.938 秒 |
| title | Example Domain |
| 最终 url | https://example.com/ |
| content | 156 字符，非空，正文内容与 Example Domain 页面一致 |
| truncated / content_type | false / text/html |
| structuredContent | 正好五个约定字段，字段类型正确 |
| 失败后恢复 | tools/list 可用，随后同一 Client/Server 连接再次公网调用成功 |
| stderr | 三条固定错误代码，没有 warning、完整 URL 或 traceback |
| MCP 日志通知 | 0 |
| 关闭 | stdin EOF 后退出码 0，SDK 强制终止调用 0 次 |

正文的脱敏短摘录：`This domain is for use in documentation examples without needing permission.` 完整的 156 字符公开正文保存在验收 JSON，未记录 query、DNS 地址或原始异常。

生产调用路径执行了系统 DNS、全候选 IP 检查、固定数字 IP TCP 连接和启用证书验证的 HTTPS 握手，随后才读取 HTML。没有 Fake Fetcher、代理回退、verify=False 或自定义信任证书。这里的 DNS/TLS 成功结论依据未替换的生产链路和真实工具成功结果；未额外采集网络包、证书指纹或完整证书链。

前期一次重复验收中 example.com 返回 TIMEOUT，保持原策略后改用公开 python.org 页面成功，标题 Welcome to Python.org，正文 5,883 字符。该瞬时超时没有进一步归因到 DNS/TCP/TLS 的具体阶段，不能称为已证明的代理限制。最终 example.com 验收及失败后重试均成功。未修改 5/10/30 秒预算以迁就网络。

### 异常场景（真实生产 stdio）

| 场景 | isError | 固定代码 |
| --- | --- | --- |
| file scheme 非法 URL | true | INVALID_URL |
| localhost（含 query canary） | true | TARGET_NOT_ALLOWED |
| 私有 IPv4 字面量 | true | TARGET_NOT_ALLOWED |

输入在安全层拒绝，无需访问内部服务。工具错误没有回显目标地址或 query。三次失败后保留同一协议会话，工具发现和再次公网抓取均成功。

### 安全复核与最小修复

复核 security.py、fetcher.py、server.py、__main__.py、HTML 解析器，以及实际固定版本 HTTPX/HTTPcore 的连接/TLS 实现：

| 边界 | 复核结论及证据 |
| --- | --- |
| DNS 与 TCP 绑定 | 原域名只用于受控解析；Transport 使用已验证数字 IP；真实 Transport 链的连接测试通过 |
| Host/SNI/TLS | 同一无尾点 ASCII 域名；SNI 使用 str；verify=True；配置/失败传播测试和真实 HTTPS 成功路径通过 |
| 重定向 | 手动处理，每跳重查；HTTPS 降级拒绝；同域重绑定与跨域连接隔离测试通过 |
| 超时/取消 | 单一总截止时间；CancelledError 不被 Exception 捕获；正文等待超时/取消关闭测试通过 |
| 字节限制 | 先拒绝压缩，使用 aiter_raw，实际字节计数和 Content-Length 提前检查；超限关闭测试通过 |
| 错误/日志 | 固定错误代码，SDK 接收安全 ToolError；异常链 canary、warning canary、真实 stderr 检查通过 |
| 生产绕过 | 无环境变量 SSRF 放宽开关、代理或 TLS 验证关闭分支；生产无测试依赖导入 |

发现并修复一处 stderr 边界：pydantic-settings 2.16.2 在 FastMCP 构造期间发出 IncompleteFieldDefinitionWarning，原入口在构造后才配置日志，且 Python warnings 默认可绕过 logging 直接输出。本次仅在构造前后建立/复核安全日志，并使用 logging.captureWarnings(True) 将 warnings 送入相同白名单出口。不修改 SDK、依赖、证书验证或 SSRF。test_logging.py 回归先实际失败，修复后通过；同时覆盖运行时含合成敏感信息的 warning。

另修复一处测试时序不确定性：原共享预算用例用 50ms 预算及 30ms sleep 推测第二跳已开始，Windows 调度可能导致首跳即超时。改为首跳立即返回重定向、第二跳等待事件，由共享截止时间取消，并额外断言只创建一次 timeout_at。保留 TIMEOUT、两次请求及 Transport 关闭断言；生产预算不变。

本轮修改：server.py、__main__.py 的最小日志修复，测试 fixture 同步调整，test_fetcher.py 的确定性修正；新增 test_logging.py、手动验收客户端及两份证据文件；同步文档。没有其他生产逻辑修复需要。

## 已知限制及未验证项

- 不宣称完全解决 SSRF。固定数字 IP 避免被检查域名的二次解析；不控制特殊路由、透明代理或网络自定义地址转换。
- Phase 3 已通过真实外站及启用证书验证的 HTTPS 成功路径；没有实际过期/伪造证书现场拒绝测试、数据包捕获或完整证书链审计。记录型 TLS 测试仍不能替代这些证据。
- 系统 DNS 超时不保证立刻终止底层线程。30 秒为协作式预算，同步解析/清理可能使墙钟时间超过截止时间。
- IP 分类随 Python 版本变化。依赖升级需要复核 SNI、地址规则和测试。
- 拒绝压缩、XHTML、备用 IP 自动重试、非默认端口、映射及特殊 IPv6，兼容性有限。
- HTMLParser 不执行 CSS/JS，不做完整浏览器可见性或复杂文章抽取；畸形 HTML 可能影响正文质量。
- SDK 的工具参数校验可能回显调用方非法值；畸形协议消息也有 SDK 自身错误/通知路径。应用日志过滤不等于整个协议栈净化，本轮没有修改 SDK 协议层。
- 没有对真实公网执行所有重定向、DNS 重绑定或超限攻击场景，这些由确定性测试验证。不宣称安全边界已被穷尽证明。
- 本轮没有运行主项目测试；独立模块没有修改或导入主项目实现。Phase 3 完成后停止，平台集成未启动。

## Git 范围

交付检查时分支 main、HEAD 059e37cc906a654e621c55730c34f1ec394e007a。全部新增文件位于 mcp_servers/web_page_fetch；现有主项目受跟踪文件无 diff。未 Commit、Push 或创建 Tag。虚拟环境被忽略。
