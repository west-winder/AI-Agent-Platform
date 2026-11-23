from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import tracing_context

from agent_loop import build_agent_graph
from scripted_model import ScriptedChatModel, weather_request


def main() -> None:
    scenarios = [
        (
            "正常完成",
            "What is the simulated weather in Beijing?",
            [weather_request(("beijing", "Beijing")), "The simulated weather in Beijing is sunny."],
        ),
        (
            "工具执行失败",
            "What is the simulated weather in Unknown?",
            [weather_request(("unknown", "Unknown")), "The lookup failed. I cannot report the weather."],
        ),
        (
            "达到模型调用上限",
            "Look up simulated weather in Beijing, Unknown, and Shanghai.",
            [
                weather_request(("beijing", "Beijing")),
                weather_request(("unknown", "Unknown")),
                weather_request(("shanghai", "Shanghai")),
            ],
        ),
    ]
    for title, question, responses in scenarios:
        model = ScriptedChatModel(messages=iter(responses), cache=False)
        with tracing_context(enabled=False):
            state = build_agent_graph(model).invoke({"messages": [HumanMessage(content=question)]})

        print(f"\n{title} | llm_calls={state['llm_calls']} | stop_reason={state['stop_reason']}")
        for message in state["messages"]:
            if isinstance(message, AIMessage) and message.tool_calls:
                print("  AIMessage tool_calls:", [call["id"] for call in message.tool_calls])
            elif isinstance(message, ToolMessage):
                print(f"  ToolMessage {message.tool_call_id} ({message.status}): {message.text}")
        print(state["final_answer"])


if __name__ == "__main__":
    main()
