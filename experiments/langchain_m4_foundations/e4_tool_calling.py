from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool


WEATHER = {"Beijing": "sunny", "Shanghai": "cloudy"}
QUESTION = "Use get_weather once to look up the city named exactly Beijing."
EXAMPLE_CALL = {
    "id": "weather-1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'},
}


@tool
def get_weather(city: str) -> str:
    """Look up simulated weather for a city by its exact name."""
    if city not in WEATHER:
        raise ValueError(f"No simulated weather for {city}")
    return WEATHER[city]


def request_weather(model: BaseChatModel, question: str) -> AIMessage:
    # bind_tools 提供工具描述与参数 Schema。invoke 返回模型的调用提议。
    return model.bind_tools([get_weather]).invoke(question)


if __name__ == "__main__":
    from mock_deepseek import fixed_completion

    print(get_weather.name, get_weather.description)
    print(get_weather.get_input_schema().model_json_schema())
    print("独立工具执行", get_weather.invoke({"city": "Beijing"}))
    with fixed_completion(
        {"content": None, "tool_calls": [EXAMPLE_CALL]}, finish_reason="tool_calls"
    ) as (model, requests):
        message = request_weather(model, QUESTION)
        print("发给模型的工具定义", requests[0]["tools"])
        print("模型提出的调用", message.tool_calls)
