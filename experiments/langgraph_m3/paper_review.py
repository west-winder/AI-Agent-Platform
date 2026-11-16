"""M3：纯本地模拟业务，真实 Send / Reducer / Subgraph。"""

import operator
import re
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class Chapter(TypedDict):
    chapter_id: str
    content: str


class Result(TypedDict):
    chapter_id: str
    status: str
    result: str


class ReferenceResult(Result):
    failure_reason: str | None


class PaperState(TypedDict, total=False):
    paper: list[Chapter]
    chapters: list[Chapter]
    content_results: Annotated[list[Result], operator.add]
    reference_results: Annotated[list[ReferenceResult], operator.add]
    events: Annotated[list[dict], operator.add]
    report: list[dict]


class ReferenceState(TypedDict, total=False):
    chapter_id: str
    content: str
    references: list[str]
    failure_reason: str | None
    output: ReferenceResult
    trace: Annotated[list[str], operator.add]


def extract_references(state: ReferenceState):
    return {"references": re.findall(r"\[ref:([^\]]+)\]", state["content"]),
            "trace": ["ExtractReferences"]}


def validate_references(state: ReferenceState):
    refs = state["references"]
    reason = "未找到引用" if not refs else (
        "模拟引用不存在：missing" if "missing" in refs else None)
    return {"failure_reason": reason, "trace": ["ValidateReferences"]}


def generate_reference_result(state: ReferenceState):
    reason = state["failure_reason"]
    return {"output": {"chapter_id": state["chapter_id"],
                       "status": "failed" if reason else "ok",
                       "result": f"extracted={len(state['references'])}",
                       "failure_reason": reason},
            "trace": ["GenerateReferenceResult"]}


def build_reference_graph():
    builder = StateGraph(ReferenceState)
    builder.add_node("ExtractReferences", extract_references)
    builder.add_node("ValidateReferences", validate_references)
    builder.add_node("GenerateReferenceResult", generate_reference_result)
    builder.add_edge(START, "ExtractReferences")
    builder.add_edge("ExtractReferences", "ValidateReferences")
    builder.add_edge("ValidateReferences", "GenerateReferenceResult")
    builder.add_edge("GenerateReferenceResult", END)
    return builder.compile()


def parse_paper(state: PaperState):
    # 复制输入；chapter_id 唯一，报告不使用任务完成顺序作为章节顺序。
    chapters = [Chapter(chapter_id=c["chapter_id"], content=c["content"])
                for c in state["paper"]]
    ids = [c["chapter_id"] for c in chapters]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("chapter_id 必须是非空且唯一的字符串")
    return {"chapters": chapters}


def dispatch_tasks(state: PaperState):
    return {"events": [{"node": "DispatchTasks", "task_count": 2 * len(state["chapters"])}]}


def send_chapters(state: PaperState):
    # 循环只构造 Send 消息，实际执行由 LangGraph Runtime 调度。
    if not state["chapters"]:
        return "GenerateReport"
    return [Send(node, dict(chapter)) for chapter in state["chapters"]
            for node in ("AnalyzeContent", "CheckReferences")]


def worker_event(node: str, chapter: Chapter, config: RunnableConfig, **extra):
    return {"node": node, "chapter_id": chapter["chapter_id"],
            "content": chapter["content"], "input_keys": sorted(chapter),
            "step": config["metadata"]["langgraph_step"], **extra}


def analyze_content(chapter: Chapter, config: RunnableConfig):
    return {"content_results": [{"chapter_id": chapter["chapter_id"], "status": "ok",
                                 "result": f"characters={len(chapter['content'])}"}],
            "events": [worker_event("AnalyzeContent", chapter, config)]}


def check_references_wrapper(subgraph, chapter: Chapter, config: RunnableConfig):
    # 子图只收到当前章节；内部字段不会自动写回主图。
    sub_input = {"chapter_id": chapter["chapter_id"], "content": chapter["content"]}
    sub_output = subgraph.invoke(sub_input, config=config)
    return {"reference_results": [sub_output["output"]],
            "events": [worker_event("CheckReferences", chapter, config,
                                    subgraph_input=sub_input,
                                    subgraph_trace=sub_output["trace"])]}


def generate_report(state: PaperState, config: RunnableConfig):
    ids = sorted(c["chapter_id"] for c in state["chapters"])
    contents = state.get("content_results", [])
    refs = state.get("reference_results", [])
    # 断言用于暴露不完整汇聚，不负责等待，也不会补跑任务。
    if sorted(r["chapter_id"] for r in contents) != ids or sorted(
            r["chapter_id"] for r in refs) != ids:
        raise RuntimeError("GenerateReport 收到不完整或重复结果")
    by_content = {r["chapter_id"]: r for r in contents}
    by_ref = {r["chapter_id"]: r for r in refs}
    return {"report": [{"chapter_id": i, "content_analysis": by_content[i],
                        "reference_check": by_ref[i]} for i in ids],
            "events": [{"node": "GenerateReport", "content_count": len(contents),
                        "reference_count": len(refs),
                        "step": config["metadata"]["langgraph_step"]}]}


def build_paper_graph():
    subgraph = build_reference_graph()

    def check_references(chapter: Chapter, config: RunnableConfig):
        return check_references_wrapper(subgraph, chapter, config)

    builder = StateGraph(PaperState)
    builder.add_node("ParsePaper", parse_paper)
    builder.add_node("DispatchTasks", dispatch_tasks)
    builder.add_node("AnalyzeContent", analyze_content)
    builder.add_node("CheckReferences", check_references)
    builder.add_node("GenerateReport", generate_report)
    builder.add_edge(START, "ParsePaper")
    builder.add_edge("ParsePaper", "DispatchTasks")
    builder.add_conditional_edges("DispatchTasks", send_chapters,
                                  ["AnalyzeContent", "CheckReferences", "GenerateReport"])
    # 两类任务都处于同一个父图 super-step；wrapper 返回前会完成整个子图。
    # 普通边在下一步触发同一个报告节点，运行证据/测试验证它只执行一次。
    builder.add_edge("AnalyzeContent", "GenerateReport")
    builder.add_edge("CheckReferences", "GenerateReport")
    builder.add_edge("GenerateReport", END)
    return builder.compile()


def mock_paper(n: int = 3) -> list[Chapter]:
    return [{"chapter_id": f"ch{i:02}",
             "content": f"Chapter {i}: local text. [ref:{'missing' if i == 2 else 'valid'}]"}
            for i in range(1, n + 1)]
