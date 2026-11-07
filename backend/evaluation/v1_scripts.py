"""Offline response scripts for the V1 dataset, separate from success criteria."""

import json

from backend.agent_runtime.contracts import LLMResponse, LLMToolCall


PROFILE = "regression-v1"


def _call(city: str, *, tool: str = "get_weather") -> LLMResponse:
    arguments = {"city": city}
    call = LLMToolCall("weather", tool, json.dumps(arguments), arguments)
    return LLMResponse(None, "tool_calls", [call])


def _answer(content: str) -> LLMResponse:
    return LLMResponse(content, "stop")


SCRIPTS = {
    "no_tool": [_answer("The capital of France is Paris.")],
    "correct_tool": [_call("Wuhan"), _answer("Wuhan is 21 C.")],
    "wrong_tool": [_call("Wuhan", tool="search_weather"), _answer("Weather unavailable.")],
    "wrong_arguments": [_call(""), _answer("The city could not be looked up.")],
    "multiple_calls": [_call("Wuhan"), _call("Beijing"), _answer("Wuhan is 21 C; Beijing is 16 C.")],
    "recover_after_tool_failure": [_call("Wuhan"), _call("Wuhan"), _answer("Wuhan is 21 C.")],
    "successful_tool_wrong_answer": [_call("Wuhan"), _answer("Wuhan is 29 C.")],
    "tool_failure_task_failure": [_call("Wuhan"), _answer("Weather unavailable.")],
    "fixture_mismatch": [_call("Beijing"), _answer("Weather unavailable.")],
    "fixture_exhausted": [_call("Wuhan"), _call("Wuhan"), _answer("The second lookup failed.")],
    "max_iterations": [_call("Wuhan"), _call("Wuhan"), _call("Wuhan")],
    "expected_runtime_exception": [_call("Wuhan")],
    "reference_contradiction": [_answer("The museum closes at 18:00 and tickets cost 20 yuan.")],
    "correct_without_tool_evidence": [_answer("Wuhan is 21 C.")],
    "judge_uncertainty": [_answer("Reservations are required on Sundays.")],
}
