import json
import platform
from importlib.metadata import version

from paper_review import build_paper_graph, mock_paper
from superstep_demo import run_superstep_demo


if __name__ == "__main__":
    print(f"Python {platform.python_version()} | langgraph {version('langgraph')} | pytest {version('pytest')}")
    state = build_paper_graph().invoke({"paper": mock_paper()}, config={"max_concurrency": 16})
    for event in state["events"]:
        print(json.dumps(event, ensure_ascii=False))
    print("REPORT")
    print(json.dumps(state["report"], ensure_ascii=False, indent=2))
    print("SUPERSTEP")
    for event in run_superstep_demo():
        print(f"step={event['step']} {event['event']}")
