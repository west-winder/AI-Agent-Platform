import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

from experiments.langchain_m4_foundations.deepseek_config import read_deepseek_config
from messages_graph import build_messages_graph


@pytest.mark.deepseek_smoke
def test_deepseek_messages_smoke():
    try:
        config = read_deepseek_config()
        if not config["DEEPSEEK_API_KEY"]:
            pytest.skip("DEEPSEEK_API_KEY is missing")
        if not config["DEFAULT_MODEL"]:
            pytest.skip("DEFAULT_MODEL is missing")

        model = ChatDeepSeek(
            model=config["DEFAULT_MODEL"],
            api_key=config["DEEPSEEK_API_KEY"],
            base_url=config["DEEPSEEK_BASE_URL"] or "https://api.deepseek.com",
            extra_body={"thinking": {"type": "disabled"}},
            timeout=20.0,
            max_retries=0,
            max_tokens=64,
            cache=False,
        )
        question = HumanMessage(content="Reply with one short greeting.", id="smoke-human")
        result = build_messages_graph(model).invoke({"messages": [question]})
    except Exception as exc:
        # SDK 异常可能携带请求或配置信息，证据只保留异常类和 HTTP 状态。
        status = getattr(exc, "status_code", None)
        detail = f" (HTTP {status})" if isinstance(status, int) else ""
        pytest.fail(f"DeepSeek smoke: {type(exc).__name__}{detail}", pytrace=False)

    messages = result["messages"]
    if len(messages) != 2 or not isinstance(messages[0], HumanMessage) or messages[0] != question:
        pytest.fail("Expected the original HumanMessage and one model reply", pytrace=False)
    reply = messages[1]
    if not isinstance(reply, AIMessage) or not reply.text.strip():
        pytest.fail("Expected an AIMessage with nonempty text", pytrace=False)

    print(f"DeepSeek messages=HumanMessage,AIMessage reply_characters={len(reply.text.strip())}")
