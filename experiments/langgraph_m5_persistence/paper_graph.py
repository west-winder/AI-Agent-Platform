import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer


class PaperInput(TypedDict):
    run_id: str
    raw_paper_content: str


class PaperUpdate(TypedDict, total=False):
    # 本轮字段在初始化和后续节点中填充，输入快照可能尚无这些值。
    paper_content: str
    analysis: str
    references: list[str]
    report: str

    # Thread 字段由 GenerateReport 在每轮完成时一起更新。
    completed_runs: int
    report_history: list[str]


class PaperState(PaperInput, PaperUpdate):
    """输入和节点更新组成的图状态。"""


def initialize_run(_state: PaperState) -> PaperUpdate:
    # 新输入已覆盖 run_id 和原文。显式清空上轮可能留在检查点中的派生字段。
    return {"paper_content": "", "analysis": "", "references": [], "report": ""}


def parse_paper(state: PaperState) -> PaperUpdate:
    return {"paper_content": " ".join(state["raw_paper_content"].split())}


def analyze_content(state: PaperState) -> PaperUpdate:
    return {"analysis": f"内容摘要｜{state['paper_content']}"}


def check_references(state: PaperState) -> PaperUpdate:
    references = re.findall(r"\[ref:([^\s\[\]]+)\]", state["paper_content"])
    return {"references": references}


def generate_report(state: PaperState) -> PaperUpdate:
    references = "、".join(state["references"]) or "无"
    report = f"{state['run_id']}｜{state['analysis']}｜引用 {references}"

    # 只有报告节点写入历史。用新列表替换旧列表，每次完成时追加一份报告。
    return {
        "report": report,
        "completed_runs": state.get("completed_runs", 0) + 1,
        "report_history": [*state.get("report_history", []), report],
    }


def build_paper_graph(
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[PaperState, None, PaperInput, PaperState]:
    graph = StateGraph(PaperState, input_schema=PaperInput)
    graph.add_node("InitializeRun", initialize_run)
    graph.add_node("ParsePaper", parse_paper)
    graph.add_node("AnalyzeContent", analyze_content)
    graph.add_node("CheckReferences", check_references)
    graph.add_node("GenerateReport", generate_report)

    graph.add_edge(START, "InitializeRun")
    graph.add_edge("InitializeRun", "ParsePaper")
    graph.add_edge("ParsePaper", "AnalyzeContent")
    graph.add_edge("ParsePaper", "CheckReferences")
    # 列表边交给 Runtime 做 fan-in，两个分支完成后才调度报告节点。
    graph.add_edge(["AnalyzeContent", "CheckReferences"], "GenerateReport")
    graph.add_edge("GenerateReport", END)
    return graph.compile(checkpointer=checkpointer)
