import json

import gradio as gr
import requests


BASE_URL = "http://127.0.0.1:8000"

USER_ID = 1


# ==================================================
# Agent
# ==================================================

def get_agents():
    response = requests.get(f"{BASE_URL}/agents")

    if response.status_code != 200:
        return {}

    agents = response.json()

    return {agent["name"]: agent["id"] for agent in agents}


AGENTS = get_agents()


# ==================================================
# Agent CRUD helpers
# ==================================================

def list_agents():
    response = requests.get(f"{BASE_URL}/agents")

    if response.status_code != 200:
        return []

    return response.json()


def refresh_agents():
    global AGENTS

    agents = list_agents()
    AGENTS = {agent["name"]: agent["id"] for agent in agents}

    return agents


def api_create_agent(name, system_prompt):
    response = requests.post(
        f"{BASE_URL}/agents",
        json={
            "name": name,
            "system_prompt": system_prompt,
        },
    )

    if response.status_code != 200:
        return None

    return response.json()


def api_update_agent(agent_id, name, system_prompt):
    response = requests.put(
        f"{BASE_URL}/agents/{agent_id}",
        json={
            "name": name,
            "system_prompt": system_prompt,
        },
    )

    if response.status_code != 200:
        return None

    return response.json()


def api_delete_agent(agent_id):
    response = requests.delete(f"{BASE_URL}/agents/{agent_id}")

    if response.status_code != 200:
        return False

    return True


# ==================================================
# Conversation
# ==================================================

def get_conversations(agent_name=None):
    """获取 conversation 列表"""
    response = requests.get(
        f"{BASE_URL}/conversations",
        params={"user_id": USER_ID},
    )

    if response.status_code != 200:
        return {}, []

    conversations = response.json()

    conversation_map = {}
    choices = []

    target_agent_id = None
    if agent_name:
        target_agent_id = AGENTS[agent_name]

    for conv in conversations:
        if target_agent_id and conv["agent_id"] != target_agent_id:
            continue

        title = conv["title"] if conv["title"] else f"未命名聊天-{conv['id']}"

        conversation_map[title] = conv["id"]
        choices.append(title)

    return conversation_map, choices


def create_conversation(agent_name):
    if agent_name is None:
        return None, get_conversations()[0], "请先选择Agent", []

    agent_id = AGENTS[agent_name]

    response = requests.post(
        f"{BASE_URL}/conversations",
        params={"user_id": USER_ID},
        json={"agent_id": agent_id},
    )

    if response.status_code != 200:
        return None, get_conversations()[0], "创建失败", []

    data = response.json()
    conversation_id = data["id"]

    conv_map, _ = get_conversations()

    return conversation_id, conv_map, f"创建成功 ID:{conversation_id}", []


def api_delete_conversation(conversation_id):
    response = requests.delete(f"{BASE_URL}/conversations/{conversation_id}")

    if response.status_code != 200:
        return False

    return True


def delete_conversation_frontend(conversation_id):
    if conversation_id is None:
        return get_conversations()[0], [], None, "请先选择聊天"

    ok = api_delete_conversation(conversation_id)
    conv_map, _ = get_conversations()

    if not ok:
        return conv_map, [], conversation_id, "删除失败"

    return conv_map, [], None, f"删除成功 ID:{conversation_id}"


# ==================================================
# Message
# ==================================================

def to_gradio_messages(messages):
    """
    Backend Message[]
        ↓
    Gradio Chatbot messages format

    当前安装的 Gradio 版本（6.x）中，
    gr.Chatbot 不再支持旧的 tuple / list pair 格式，
    只接受：

        [
            {
                "role": "user",
                "content": "..."
            },
            {
                "role": "assistant",
                "content": "..."
            }
        ]

    这是整个 Gradio Chatbot 边界的
    唯一转换入口。

    UI 只展示：

        user
        assistant

    其余 role（如果后端未来出现）
    不在这里发明转换行为，
    直接跳过。
    """

    gradio_messages = []

    for message in messages or []:

        if not isinstance(message, dict):
            continue

        role = message.get("role")

        content = message.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str):
            continue

        gradio_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return gradio_messages


def load_messages(title, conversation_map):
    if title is None:
        return [], None

    conversation_id = conversation_map[title]

    response = requests.get(
        f"{BASE_URL}/messages",
        params={"conversation_id": conversation_id},
    )

    if response.status_code != 200:
        return [], conversation_id

    messages = response.json()

    return to_gradio_messages(messages), conversation_id


# ==================================================
# Chat
# ==================================================

def chat(message, history, conversation_id, stream_enabled=True):
    """
    发送一条用户消息。

    这是一个 Generator Function。

    因为 Streaming 分支必须使用 yield，
    而一旦函数体内出现 yield，
    整个函数就变成 Generator Contract，
    所以 Non-Streaming 分支也必须 yield，
    不能使用 return 作为 Gradio 输出。

    两个分支的输出 Contract 完全一致：

        (gradio_messages, "")

    gradio_messages 与 load_messages() 使用同一个
    Gradio messages Contract，
    不能混用 tuple / list pair 格式。
    """

    gradio_messages = to_gradio_messages(history)

    # --------------------------------------------------
    # 1. conversation_id 检查
    # --------------------------------------------------

    if conversation_id is None:

        gradio_messages.append(
            {
                "role": "assistant",
                "content": "请先创建聊天",
            }
        )

        yield gradio_messages, ""

        return

    # --------------------------------------------------
    # 2. 空消息直接忽略
    #
    # 否则会把 content=None 写进 Chatbot messages，
    # 破坏 messages Contract。
    # --------------------------------------------------

    if not message:

        yield gradio_messages, ""

        return

    # --------------------------------------------------
    # 3. Non-Streaming 分支
    #
    # stream_enabled == False 时：
    #
    #   POST /chat
    #   → response.json()["answer"]
    #   → 一次性展示完整 Assistant Message
    #
    # 只 yield 一次，UI 表现与旧版本一致。
    # --------------------------------------------------

    if not stream_enabled:

        response = None

        try:

            response = requests.post(
                f"{BASE_URL}/chat",
                json={
                    "conversation_id": conversation_id,
                    "content": message,
                },
            )

        except requests.RequestException:

            response = None

        if response is None or response.status_code != 200:

            gradio_messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )
            gradio_messages.append(
                {
                    "role": "assistant",
                    "content": "请求失败",
                }
            )

            yield gradio_messages, ""

            return

        data = response.json()
        answer = data["answer"]

        gradio_messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        gradio_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        yield gradio_messages, ""

        return

    # --------------------------------------------------
    # 4. Streaming 分支
    #
    # stream_enabled == True 时：
    #
    #   POST /chat/stream
    #   → SSE
    #   → 每收到 data: {"delta": "..."}
    #   → 更新“同一个”Assistant Message
    # --------------------------------------------------

    gradio_messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    gradio_messages.append(
        {
            "role": "assistant",
            "content": "",
        }
    )

    # 先 yield 一次：
    # 让 User Message 与空 Assistant Message 立即出现在页面上。
    yield gradio_messages, ""

    assistant_text = ""

    try:

        with requests.post(
            f"{BASE_URL}/chat/stream",
            json={
                "conversation_id": conversation_id,
                "content": message,
            },
            headers={
                "Accept": "text/event-stream",
            },
            stream=True,
        ) as response:

            if response.status_code != 200:

                gradio_messages[-1]["content"] = "请求失败"

                yield gradio_messages, ""

                return

            # decode_unicode=False：
            # 自己按 UTF-8 解码，
            # 不依赖 requests 从 Content-Type 推断编码，
            # 保证中文不乱码。
            for raw_line in response.iter_lines(decode_unicode=False):

                # SSE 事件之间用空行分隔，
                # 空行不是正文，直接跳过。
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="replace")

                # 只处理 data: 行
                if not line.startswith("data:"):
                    continue

                payload = line[len("data:"):].strip()

                if not payload:
                    continue

                # 当前 data 内容是 JSON：{"delta": "..."}
                try:

                    chunk_data = json.loads(payload)

                except json.JSONDecodeError:

                    # 单个 chunk 解析失败不中断整条流
                    continue

                delta = chunk_data.get("delta")

                if not isinstance(delta, str) or not delta:
                    continue

                assistant_text += delta

                # 始终更新最后一个 Assistant Message，
                # 绝不 append 新的 assistant message。
                gradio_messages[-1]["content"] = assistant_text

                yield gradio_messages, ""

    except requests.RequestException:

        # Streaming 中途失败：
        # 保留已收到的内容，追加中断提示，
        # 不让整个 Gradio 页面崩掉。
        if assistant_text:

            gradio_messages[-1]["content"] = (
                f"{assistant_text}\n\n[流式请求中断]"
            )

        else:

            gradio_messages[-1]["content"] = "流式请求中断"

        yield gradio_messages, ""

        return

    # --------------------------------------------------
    # 5. Streaming 结束兜底
    #
    # 如果整条流没有任何有效 delta，
    # 空 Assistant Message 会被替换成提示文案，
    # 避免留下一个永远为空的 Assistant 气泡。
    # --------------------------------------------------

    if not assistant_text:

        gradio_messages[-1]["content"] = "（无内容返回）"

        yield gradio_messages, ""


# ==================================================
# Agent management event handlers
# ==================================================

def select_agent_for_manage(agent_name, agents):
    agents = agents or []

    if not agent_name:
        return "", "", None

    for agent in agents:
        if agent["name"] == agent_name:
            return agent["name"], agent["system_prompt"], agent["id"]

    return agent_name, "", None


def create_agent_frontend(name, system_prompt):
    if not name or not system_prompt:
        agents = refresh_agents()
        choices = [agent["name"] for agent in agents]
        return (
            agents,
            gr.update(choices=choices),
            gr.update(choices=choices),
            "",
            "",
            "名称和 system_prompt 不能为空",
        )

    result = api_create_agent(name, system_prompt)
    if result is None:
        agents = refresh_agents()
        choices = [agent["name"] for agent in agents]
        return (
            agents,
            gr.update(choices=choices),
            gr.update(choices=choices),
            "",
            "",
            "创建失败",
        )

    agents = refresh_agents()
    choices = [agent["name"] for agent in agents]
    return (
        agents,
        gr.update(choices=choices, value=name),
        gr.update(choices=choices),
        "",
        "",
        f"创建成功 ID:{result['id']}",
    )


def update_agent_frontend(agent_id, name, system_prompt):
    if agent_id is None:
        agents = refresh_agents()
        choices = [agent["name"] for agent in agents]
        return (
            agents,
            gr.update(choices=choices),
            gr.update(choices=choices),
            "",
            "",
            "请先选择Agent",
        )

    if not name or not system_prompt:
        agents = refresh_agents()
        choices = [agent["name"] for agent in agents]
        return (
            agents,
            gr.update(choices=choices),
            gr.update(choices=choices),
            name,
            system_prompt,
            "名称和 system_prompt 不能为空",
        )

    result = api_update_agent(agent_id, name, system_prompt)
    if result is None:
        agents = refresh_agents()
        choices = [agent["name"] for agent in agents]
        return (
            agents,
            gr.update(choices=choices),
            gr.update(choices=choices),
            name,
            system_prompt,
            "更新失败",
        )

    agents = refresh_agents()
    choices = [agent["name"] for agent in agents]
    return (
        agents,
        gr.update(choices=choices, value=name),
        gr.update(choices=choices),
        name,
        system_prompt,
        f"更新成功 ID:{agent_id}",
    )


def delete_agent_frontend(agent_id, name, system_prompt):
    if agent_id is None:
        agents = refresh_agents()
        choices = [agent["name"] for agent in agents]
        return (
            agents,
            gr.update(choices=choices),
            gr.update(choices=choices),
            None,
            "",
            "",
            "请先选择Agent",
        )

    ok = api_delete_agent(agent_id)
    if not ok:
        agents = refresh_agents()
        choices = [agent["name"] for agent in agents]
        return (
            agents,
            gr.update(choices=choices, value=name),
            gr.update(choices=choices),
            agent_id,
            name,
            system_prompt,
            "删除失败",
        )

    agents = refresh_agents()
    choices = [agent["name"] for agent in agents]
    return (
        agents,
        gr.update(choices=choices, value=None),
        gr.update(choices=choices),
        None,
        "",
        "",
        f"删除成功 ID:{agent_id}",
    )


# ==================================================
# UI
# ==================================================

INITIAL_CONVERSATION_MAP, _ = get_conversations()
INITIAL_AGENT_LIST = refresh_agents()


CONVERSATION_LIST_CSS = """
#conversation-list {
    max-height: 420px;
    overflow-y: auto;
}
"""


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AI Agent Chat

        FastAPI + DeepSeek Agent 系统
        """
    )

    # 状态保存
    conversation_id = gr.State(None)
    conversation_map = gr.State(INITIAL_CONVERSATION_MAP)

    with gr.Tabs():
        # =================
        # Tab1: Agent管理
        # =================
        with gr.Tab("Agent管理"):
            agents_full_state = gr.State(INITIAL_AGENT_LIST)
            selected_agent_id_state = gr.State(None)

            gr.Markdown("### 已有 Agent")
            agent_manage_select = gr.Dropdown(
                choices=[agent["name"] for agent in INITIAL_AGENT_LIST],
                label="选择Agent",
                value=None,
            )

            agent_name_text = gr.Textbox(
                label="Agent名称",
                interactive=True,
            )
            agent_prompt_text = gr.Textbox(
                label="system_prompt",
                lines=8,
                interactive=True,
            )

            with gr.Row():
                edit_btn = gr.Button("编辑Agent", variant="secondary")
                delete_btn = gr.Button("删除Agent", variant="stop")

            manage_status = gr.Textbox(label="操作状态", interactive=False)

            gr.Markdown("### 创建 Agent")
            create_name = gr.Textbox(label="name")
            create_prompt = gr.Textbox(label="system_prompt", lines=8)
            create_btn = gr.Button("创建Agent", variant="primary")

        # =================
        # Tab2: 聊天
        # =================
        with gr.Tab("聊天"):
            with gr.Row():
                # 左侧 Sidebar
                with gr.Column(scale=1):
                    agent_select = gr.Dropdown(
                        choices=list(AGENTS.keys()),
                        label="选择Agent",
                    )

                    new_chat_btn = gr.Button("新建聊天", variant="primary")
                    delete_chat_btn = gr.Button("删除当前聊天", variant="stop")

                    with gr.Column(elem_id="conversation-list", scale=1):
                        @gr.render(inputs=conversation_map)
                        def render_conversations(conv_map):
                            conv_map = conv_map or {}
                            for title in conv_map:
                                conv_btn = gr.Button(title, size="sm")
                                conv_btn.click(
                                    fn=load_messages,
                                    inputs=[conv_btn, conversation_map],
                                    outputs=[chatbot, conversation_id],
                                )

                    status = gr.Textbox(label="状态", interactive=False)

                # 右侧聊天区
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="聊天记录",
                        height=600,
                        render_markdown=True,
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$", "right": "$", "display": False},
                            {"left": "\\(", "right": "\\)", "display": False},
                            {"left": "\\[", "right": "\\]", "display": True},
                        ],
                    )
                    message = gr.Textbox(label="输入消息")

                    with gr.Row():
                        stream_enabled = gr.Checkbox(
                            label="流式输出",
                            value=True,
                        )
                        send_btn = gr.Button("发送", variant="primary")

    # Agent管理事件
    agent_manage_select.change(
        fn=select_agent_for_manage,
        inputs=[agent_manage_select, agents_full_state],
        outputs=[agent_name_text, agent_prompt_text, selected_agent_id_state],
    )

    create_btn.click(
        fn=create_agent_frontend,
        inputs=[create_name, create_prompt],
        outputs=[
            agents_full_state,
            agent_manage_select,
            agent_select,
            create_name,
            create_prompt,
            manage_status,
        ],
    )

    edit_btn.click(
        fn=update_agent_frontend,
        inputs=[selected_agent_id_state, agent_name_text, agent_prompt_text],
        outputs=[
            agents_full_state,
            agent_manage_select,
            agent_select,
            agent_name_text,
            agent_prompt_text,
            manage_status,
        ],
    )

    delete_btn.click(
        fn=delete_agent_frontend,
        inputs=[selected_agent_id_state, agent_name_text, agent_prompt_text],
        outputs=[
            agents_full_state,
            agent_manage_select,
            agent_select,
            selected_agent_id_state,
            agent_name_text,
            agent_prompt_text,
            manage_status,
        ],
    )

    # 创建 conversation，并自动切换到新会话
    new_chat_btn.click(
        fn=create_conversation,
        inputs=[agent_select],
        outputs=[conversation_id, conversation_map, status, chatbot],
    )

    delete_chat_btn.click(
        fn=delete_conversation_frontend,
        inputs=[conversation_id],
        outputs=[conversation_map, chatbot, conversation_id, status],
    )

    # 发送消息
    send_btn.click(
        fn=chat,
        inputs=[message, chatbot, conversation_id, stream_enabled],
        outputs=[chatbot, message],
    )


if __name__ == "__main__":
    # Gradio 4.44.1 + gradio_client 1.3.0 cannot parse JSON Schemas that
    # contain a boolean `additionalProperties` (e.g. Chatbot).  Guard that
    # case so the root route can generate its API info instead of raising
    # "argument of type 'bool' is not iterable".
    import gradio_client.utils as _gradio_client_utils

    _original_json_schema_to_python_type = (
        _gradio_client_utils._json_schema_to_python_type
    )

    def _json_schema_to_python_type(schema, defs):
        if not isinstance(schema, dict):
            return "Any"
        return _original_json_schema_to_python_type(schema, defs)

    _gradio_client_utils._json_schema_to_python_type = (
        _json_schema_to_python_type
    )

    # Starlette 1.x removed the old TemplateResponse(name, context) calling
    # convention that Gradio 4.44.1 still uses.  Translate the old call into
    # TemplateResponse(request, name, context).
    import gradio.routes as _gradio_routes

    _original_template_response = _gradio_routes.templates.TemplateResponse

    def _template_response(*args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else {}
            request = context.get("request") if isinstance(context, dict) else None
            return _original_template_response(request, name, context, **kwargs)
        return _original_template_response(*args, **kwargs)

    _gradio_routes.templates.TemplateResponse = _template_response

    # Gradio 6：css 必须传给 launch()，
    # 不能再传给 Blocks()。
    demo.launch(
        css=CONVERSATION_LIST_CSS
    )
