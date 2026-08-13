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

    history = []
    temp = []

    for msg in messages:
        temp.append(msg)

    # 数据库格式:
    # user
    # assistant
    # 转换为 Gradio 格式:
    # [["问题", "回答"]]
    for i in range(0, len(temp), 2):
        if i + 1 < len(temp):
            history.append([temp[i]["content"], temp[i + 1]["content"]])

    return history, conversation_id


# ==================================================
# Chat
# ==================================================

def chat(message, history, conversation_id):
    if history is None:
        history = []

    if conversation_id is None:
        history.append(["", "请先创建聊天"])
        return history, ""

    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "conversation_id": conversation_id,
            "content": message,
        },
    )

    if response.status_code != 200:
        history.append([message, "请求失败"])
        return history, ""

    data = response.json()
    answer = data["answer"]

    history.append([message, answer])
    return history, ""


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


with gr.Blocks(
    css="""
    #conversation-list {
        max-height: 420px;
        overflow-y: auto;
    }
    """
) as demo:
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
        inputs=[message, chatbot, conversation_id],
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

    demo.launch()
