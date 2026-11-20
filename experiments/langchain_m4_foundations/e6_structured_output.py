import json
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, ValidationError


class PaperDecision(BaseModel):
    model_config = ConfigDict(strict=True)

    paper_title: str
    needs_review: bool
    reason: str


class StructuredResult(TypedDict):
    raw: BaseMessage
    parsed: PaperDecision | None
    parsing_error: BaseException | None


EXAMPLE = {
    "paper_title": "A Study of Graphs",
    "needs_review": True,
    "reason": "The abstract omits the evaluation method.",
}
INVALID = {**EXAMPLE, "needs_review": "yes"}
QUESTION = (
    "Return a JSON object for the paper 'A Study of Graphs'. "
    "Its abstract omits the evaluation method, so flag it for review. "
    f"Use this JSON shape: {json.dumps(EXAMPLE)}"
)


def review_paper(model: BaseChatModel, question: str) -> StructuredResult:
    # json_mode 约束服务端格式。适配器在客户端用 PaperDecision 校验字段。
    return model.with_structured_output(
        PaperDecision, method="json_mode", include_raw=True
    ).invoke(question)


if __name__ == "__main__":
    from mock_deepseek import fixed_completion

    print("Schema", PaperDecision.model_json_schema())
    decision = PaperDecision.model_validate(EXAMPLE)
    print("本地校验", decision.paper_title, decision.needs_review, decision.reason)
    try:
        PaperDecision.model_validate(INVALID)
    except ValidationError as exc:
        print("本地错误", exc.errors(include_input=False))

    for data in (EXAMPLE, INVALID):
        with fixed_completion({"content": json.dumps(data)}) as (model, requests):
            result = review_paper(model, QUESTION)
            print("response_format", requests[0]["response_format"])
            print("raw", type(result["raw"]).__name__, result["raw"].content)
            print("parsed", result["parsed"])
            print("parsing_error", type(result["parsing_error"]).__name__)
