import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.database import SessionLocal

from backend.memory.memory_read.memory_reader import (
    MemoryReader
)

from backend.memory.memory_read.dense_retriever import (
    DenseRetrievedMemory
)

from backend.memory.memory_read.memory_relevance_judge import (
    JudgeDecision
)

from backend.retrieval.bm25 import (
    BM25Result
)

from backend.retrieval.rrf import (
    reciprocal_rank_fusion
)

from backend.reranking.reranker import (
    RerankResult
)


# ============================================================
# Test Configuration
# ============================================================

REAL_USER_ID = 1

NO_MEMORY_USER_ID = 999999

TOP_N = 20
TOP_K = 5


# ============================================================
# Print Helpers
# ============================================================


def format_score(score):
    """
    Runtime Trace 中统一格式化 score。
    """

    if score is None:
        return "None"

    return f"{score:.6f}"


def print_result(
    case_name,
    query,
    result
):
    """
    打印一次完整 Memory Read V2 Runtime Trace。
    """

    print()
    print("=" * 80)
    print(case_name)
    print("=" * 80)

    print()
    print("Query:")
    print(query)

    print()
    print("-" * 80)
    print("FINAL RERANKER RANKING")
    print("-" * 80)

    if not result.items:
        print("[EMPTY] No MemoryReadItem")
    else:
        for rank, item in enumerate(
            result.items,
            start=1
        ):
            print()

            print(
                f"Rank {rank}"
                f" | Memory ID {item.memory.id}"
                f" | Type {item.memory.memory_type}"
            )

            print(
                f"Content: {item.memory.content}"
            )

            print(
                "Dense: "
                f"{format_score(item.dense_score)}"
            )

            print(
                "BM25: "
                f"{format_score(item.bm25_score)}"
            )

            print(
                "RRF: "
                f"{format_score(item.rrf_score)}"
            )

            print(
                "Rerank: "
                f"{format_score(item.rerank_score)}"
            )

            print(
                "Judge Selected: "
                f"{item.judge_selected}"
            )

            print(
                "Judge Reason: "
                f"{item.judge_reason}"
            )

    print()
    print("-" * 80)
    print("SELECTED MEMORIES")
    print("-" * 80)

    selected_items = [
        item
        for item in result.items
        if item.judge_selected is True
    ]

    if not selected_items:
        print("[EMPTY] No selected memory")
    else:
        for item in selected_items:
            print(
                f"Memory ID {item.memory.id}: "
                f"{item.memory.content}"
            )

    print()
    print("-" * 80)
    print("MEMORY CONTEXT")
    print("-" * 80)

    if result.memory_context:
        print(result.memory_context)
    else:
        print("[EMPTY]")


# ============================================================
# Case 1
# LangGraph Lexical
# ============================================================


def test_langgraph_lexical(
    reader,
    db
):
    """
    明确包含 LangGraph 关键词。

    主要验证：

    SQLite Repository
        ↓
    Dense
        +
    BM25 lexical signal
        ↓
    RRF
        ↓
    Reranker
        ↓
    Judge
        ↓
    Injector
    """

    case_name = (
        "CASE 1 - LANGGRAPH LEXICAL"
    )

    query = (
        "我最近是不是在学习 LangGraph？"
    )

    result = reader.read(
        db=db,
        user_id=REAL_USER_ID,
        query=query,
        top_n=TOP_N,
        top_k=TOP_K
    )

    print_result(
        case_name=case_name,
        query=query,
        result=result
    )

    # --------------------------------------------------------
    # 1. 必须存在候选
    # --------------------------------------------------------

    assert result.items, (
        "LangGraph Lexical Case "
        "没有返回任何 Candidate"
    )

    # --------------------------------------------------------
    # 2. ID 1 / ID 2 至少一个必须进入最终 Candidate
    # --------------------------------------------------------

    langgraph_items = [
        item
        for item in result.items
        if item.memory.id in {1, 2}
    ]

    assert langgraph_items, (
        "LangGraph Memory "
        "没有进入最终 Candidate"
    )

    # --------------------------------------------------------
    # 3. BM25 必须对至少一个 LangGraph Memory
    #    给出正 lexical score
    # --------------------------------------------------------

    lexical_hits = [
        item
        for item in langgraph_items
        if (
            item.bm25_score is not None
            and item.bm25_score > 0
        )
    ]

    assert lexical_hits, (
        "LangGraph Lexical Case 中，"
        "BM25 没有产生有效 lexical signal"
    )

    # --------------------------------------------------------
    # 4. Judge 至少选择一个 LangGraph Memory
    # --------------------------------------------------------

    selected_langgraph = [
        item
        for item in langgraph_items
        if item.judge_selected is True
    ]

    assert selected_langgraph, (
        "Judge 没有选择任何 "
        "LangGraph 相关 Memory"
    )

    # --------------------------------------------------------
    # 5. 最终 Context 必须真正包含 LangGraph
    # --------------------------------------------------------

    assert (
        "LangGraph"
        in result.memory_context
    ), (
        "memory_context 中没有 "
        "LangGraph Memory"
    )

    print()
    print(
        "[PASS] CASE 1 - LANGGRAPH LEXICAL"
    )


# ============================================================
# Case 2
# LangGraph Semantic
# ============================================================


def test_langgraph_semantic(
    reader,
    db
):
    """
    Query 不直接写 LangGraph。

    主要观察 Dense 是否能够通过语义
    找到 LangGraph / Agent 工作流相关 Memory。

    注意：

    这里不强制要求 BM25 == 0。

    因为 query 和 Memory 之间仍可能存在
    一些普通词汇重叠。

    我们真正验证的是：

    没有 LangGraph 这个 identifier，
    仍然能召回 LangGraph Memory。
    """

    case_name = (
        "CASE 2 - LANGGRAPH SEMANTIC"
    )

    query = (
        "我最近在研究什么 Agent 编排相关的技术？"
    )

    result = reader.read(
        db=db,
        user_id=REAL_USER_ID,
        query=query,
        top_n=TOP_N,
        top_k=TOP_K
    )

    print_result(
        case_name=case_name,
        query=query,
        result=result
    )

    # --------------------------------------------------------
    # 1. 必须有候选
    # --------------------------------------------------------

    assert result.items, (
        "LangGraph Semantic Case "
        "没有返回 Candidate"
    )

    # --------------------------------------------------------
    # 2. ID 1 / ID 2 至少一个必须被召回
    # --------------------------------------------------------

    langgraph_items = [
        item
        for item in result.items
        if item.memory.id in {1, 2}
    ]

    assert langgraph_items, (
        "Semantic Query 没有召回 "
        "LangGraph Memory"
    )

    # --------------------------------------------------------
    # 3. Dense 必须真的提供 score
    # --------------------------------------------------------

    dense_hits = [
        item
        for item in langgraph_items
        if item.dense_score is not None
    ]

    assert dense_hits, (
        "LangGraph Memory 没有 "
        "Dense Retrieval Trace"
    )

    # --------------------------------------------------------
    # 4. Judge 至少选择一个 LangGraph Memory
    # --------------------------------------------------------

    selected_langgraph = [
        item
        for item in langgraph_items
        if item.judge_selected is True
    ]

    assert selected_langgraph, (
        "Semantic Query 下 Judge "
        "没有选择 LangGraph Memory"
    )

    # --------------------------------------------------------
    # 5. Context 必须包含 LangGraph
    # --------------------------------------------------------

    assert (
        "LangGraph"
        in result.memory_context
    ), (
        "Semantic Query 最终 Context "
        "没有 LangGraph"
    )

    print()
    print(
        "[PASS] CASE 2 - LANGGRAPH SEMANTIC"
    )


# ============================================================
# Case 3
# No Relevant Memory
# ============================================================


def test_no_relevant_memory(
    reader,
    db
):
    """
    数据库有 Memory，
    但是没有饮食信息。

    这里非常重要：

    Retriever 仍然可以返回 Top-N。

    我们不是要求：

        Retrieval = []

    而是要求：

        Candidate
            ↓
        Reranker
            ↓
        Judge
            ↓
        selected = []
            ↓
        memory_context = ""
    """

    case_name = (
        "CASE 3 - NO RELEVANT MEMORY"
    )

    query = (
        "我最喜欢吃什么？"
    )

    result = reader.read(
        db=db,
        user_id=REAL_USER_ID,
        query=query,
        top_n=TOP_N,
        top_k=TOP_K
    )

    print_result(
        case_name=case_name,
        query=query,
        result=result
    )

    # --------------------------------------------------------
    # 数据库明明有 8 条 Memory，
    # 所以 Retrieval 阶段通常仍应有 Candidate。
    # --------------------------------------------------------

    assert result.items, (
        "No Relevant Memory Case "
        "完全没有 Candidate，"
        "请检查 Retrieval 是否异常"
    )

    # --------------------------------------------------------
    # Judge 不应该选择无关 Memory
    # --------------------------------------------------------

    selected_items = [
        item
        for item in result.items
        if item.judge_selected is True
    ]

    assert not selected_items, (
        "Judge 错误选择了与饮食问题"
        "无关的 Memory"
    )

    # --------------------------------------------------------
    # 最终不能注入 Memory Context
    # --------------------------------------------------------

    assert not result.memory_context.strip(), (
        "No Relevant Memory Case "
        "仍然生成了 memory_context"
    )

    print()
    print(
        "[PASS] CASE 3 - NO RELEVANT MEMORY"
    )


# ============================================================
# Case 4
# No Memory
# ============================================================


def test_no_memory(
    reader,
    db
):
    """
    Repository 返回 []。

    这是合法空结果，
    不是 Failure。
    """

    case_name = (
        "CASE 4 - NO MEMORY"
    )

    query = (
        "我最近在学习什么？"
    )

    result = reader.read(
        db=db,
        user_id=NO_MEMORY_USER_ID,
        query=query,
        top_n=TOP_N,
        top_k=TOP_K
    )

    print_result(
        case_name=case_name,
        query=query,
        result=result
    )

    assert result.items == [], (
        "不存在 Memory 的 user_id "
        "仍然返回了 items"
    )

    assert (
        result.memory_context == ""
    ), (
        "不存在 Memory 的 user_id "
        "仍然产生了 memory_context"
    )

    print()
    print(
        "[PASS] CASE 4 - NO MEMORY"
    )


# ============================================================
# Case 5
# top_n larger than corpus size
#
# Controlled Boundary Regression Test
#
# 不使用：
#   SQLite
#   真实 Embedding
#   真实 BM25
#   真实 Reranker
#   DeepSeek
# ============================================================


@dataclass
class BoundaryFakeMemory:
    """
    Controlled Test 使用的 Memory。

    只需要满足：
    MemoryReader 使用的 .content
    +
    print_result 使用的 .id / .memory_type
    """

    id: int
    content: str
    memory_type: str = "semantic"


class BoundaryFakeRepository:
    """
    固定返回 3 条 Memory。

    记录调用次数，
    用于确认 Repository 只被调用一次。
    """

    def __init__(
        self,
        memories
    ):
        self._memories = list(
            memories
        )

        self.call_count = 0

    def __call__(
        self,
        db,
        user_id
    ):
        self.call_count += 1

        return list(
            self._memories
        )


class BoundaryFakeDenseRetriever:
    """
    Fake Dense Retriever。

    核心职责：
    记录自己实际收到的 top_n。

    同时返回 top_n 条
    结构合法的 DenseRetrievedMemory，
    让 Pipeline 能继续往下走。
    """

    def __init__(self):
        self.call_count = 0
        self.received_top_n = None
        self.received_memory_count = None

    def search(
        self,
        query,
        memories,
        top_n=5
    ):
        self.call_count += 1

        self.received_top_n = top_n

        self.received_memory_count = len(
            memories
        )

        count = min(
            top_n,
            len(memories)
        )

        return [
            DenseRetrievedMemory(
                index=index,
                memory=memories[index],
                similarity=(
                    1.0
                    - index * 0.1
                )
            )
            for index in range(
                count
            )
        ]


class BoundaryFakeBM25Retriever:
    """
    Fake BM25 Retriever。

    核心职责：
    记录自己实际收到的 top_n。

    注意：
    score 必须 > 0，
    否则 memory_reader 会把该结果
    从 bm25_ranking 中过滤掉。
    """

    def __init__(self):
        self.index_call_count = 0
        self.search_call_count = 0
        self.received_top_n = None
        self.received_texts = None

    def index(
        self,
        texts
    ):
        self.index_call_count += 1

        self.received_texts = list(
            texts
        )

    def search(
        self,
        query,
        top_n=5
    ):
        self.search_call_count += 1

        self.received_top_n = top_n

        texts = self.received_texts

        count = min(
            top_n,
            len(texts)
        )

        return [
            BM25Result(
                index=index,
                text=texts[index],
                score=float(
                    len(texts)
                    - index
                )
            )
            for index in range(
                count
            )
        ]


class BoundaryFakeRRF:
    """
    Fake RRF。

    记录自己实际收到的 top_n，
    然后委托给真实 RRF 公式。

    RRF 是纯函数，
    不加载模型、不访问网络、不访问数据库。
    """

    def __init__(self):
        self.call_count = 0
        self.received_top_n = None
        self.received_rankings = None

    def __call__(
        self,
        rankings,
        top_n=None,
        rank_constant=60
    ):
        self.call_count += 1

        self.received_top_n = top_n

        self.received_rankings = [
            list(ranking)
            for ranking in rankings
        ]

        return reciprocal_rank_fusion(
            rankings=rankings,
            rank_constant=(
                rank_constant
            ),
            top_n=top_n
        )


class BoundaryFakeReranker:
    """
    Fake Reranker。

    不加载 Cross Encoder。

    返回覆盖全部 Candidate 的
    RerankResult。
    """

    def __init__(self):
        self.call_count = 0
        self.received_texts = None

    def rerank(
        self,
        query,
        texts
    ):
        self.call_count += 1

        self.received_texts = list(
            texts
        )

        return [
            RerankResult(
                index=index,
                text=text,
                score=float(
                    len(texts)
                    - index
                )
            )
            for index, text in enumerate(
                texts
            )
        ]


class BoundaryFakeJudge:
    """
    Fake Judge。

    不调用 DeepSeek。

    记录自己实际收到的 texts 数量。
    """

    def __init__(self):
        self.call_count = 0
        self.received_texts = None

    def judge(
        self,
        query,
        texts
    ):
        self.call_count += 1

        self.received_texts = list(
            texts
        )

        return [
            JudgeDecision(
                index=index,
                selected=True,
                reason=(
                    "boundary fake accept"
                )
            )
            for index in range(
                len(texts)
            )
        ]


class BoundaryFakeInjector:
    """
    Fake Injector。

    记录最终进入 Context 的文本。
    """

    def __init__(self):
        self.received_texts = None

    def build_context(
        self,
        texts
    ):
        self.received_texts = list(
            texts
        )

        if not texts:
            return ""

        return (
            "<memory_context>"
            + "|".join(texts)
            + "</memory_context>"
        )


def test_top_n_larger_than_corpus():
    """
    top_n > corpus size

    属于合法 Boundary，
    不是 Failure。

    Repository 返回 3 条 Memory，
    调用方请求：

        top_n = 20
        top_k = 5

    期望：

    MemoryReader 不抛异常，
    并且把 top_n 收敛为：

        effective_top_n
            = min(20, 3)
            = 3

    即：

    Dense Retriever 收到 top_n = 3
    BM25 Retriever  收到 top_n = 3
    RRF             收到 top_n = 3

    回归背景：

    修复前，
    top_n=20 被直接传给 BM25，
    bm25s 抛出：

        ValueError:
        k of 20 is larger than the number
        of available scores, which is 8
    """

    case_name = (
        "CASE 5 - TOP_N LARGER THAN CORPUS"
    )

    query = (
        "我最近在学习什么？"
    )

    # --------------------------------------------------------
    # Boundary Configuration
    # --------------------------------------------------------

    requested_top_n = 20
    requested_top_k = 5
    corpus_size = 3

    expected_effective_top_n = min(
        requested_top_n,
        corpus_size
    )

    print()
    print("-" * 80)
    print("BOUNDARY CONFIGURATION")
    print("-" * 80)

    print(
        f"Corpus Size        : "
        f"{corpus_size}"
    )

    print(
        f"Requested top_n    : "
        f"{requested_top_n}"
    )

    print(
        f"Requested top_k    : "
        f"{requested_top_k}"
    )

    print(
        f"Expected effective : "
        f"{expected_effective_top_n}"
    )

    # --------------------------------------------------------
    # Controlled Corpus
    # --------------------------------------------------------

    memories = [
        BoundaryFakeMemory(
            id=101,
            content=(
                "用户最近开始学习 LangGraph"
            )
        ),
        BoundaryFakeMemory(
            id=102,
            content=(
                "Memory Read V1 对应 "
                "Git commit 为 22ecfec"
            )
        ),
        BoundaryFakeMemory(
            id=103,
            content=(
                "用户当前使用 "
                "Qwen3-Embedding-0.6B "
                "作为 Embedding 模型"
            )
        ),
    ]

    # --------------------------------------------------------
    # Fake Components
    # --------------------------------------------------------

    repository = BoundaryFakeRepository(
        memories
    )

    dense = BoundaryFakeDenseRetriever()

    bm25 = BoundaryFakeBM25Retriever()

    rrf = BoundaryFakeRRF()

    reranker = BoundaryFakeReranker()

    judge = BoundaryFakeJudge()

    injector = BoundaryFakeInjector()

    # --------------------------------------------------------
    # MemoryReader
    #
    # 全部核心组件都是 Fake。
    #
    # 不使用：
    #   SQLite
    #   真实 Embedding
    #   真实 BM25
    #   真实 Reranker
    #   DeepSeek
    # --------------------------------------------------------

    reader = MemoryReader(
        repository_callable=repository,
        dense_retriever=dense,
        bm25_retriever=bm25,
        rrf_callable=rrf,
        reranker=reranker,
        judge=judge,
        injector=injector,
    )

    # --------------------------------------------------------
    # Execute
    #
    # 这里不捕获异常。
    #
    # "MemoryReader 不抛异常"
    # 本身就是断言的一部分。
    # --------------------------------------------------------

    result = reader.read(
        db=None,
        user_id=1,
        query=query,
        top_n=requested_top_n,
        top_k=requested_top_k
    )

    print_result(
        case_name=case_name,
        query=query,
        result=result
    )

    # --------------------------------------------------------
    # 1. Repository 返回 3 条 Memory
    # --------------------------------------------------------

    assert (
        len(memories)
        == corpus_size
    ), (
        "Controlled Corpus 构造错误，"
        "必须是 3 条 Memory"
    )

    assert (
        repository.call_count == 1
    ), (
        "Repository 应该只被调用一次，"
        f"实际 {repository.call_count} 次"
    )

    # --------------------------------------------------------
    # 2. effective_top_n 必须等于 3
    #
    # 这是本次 Regression Test 的核心断言。
    # --------------------------------------------------------

    assert (
        expected_effective_top_n
        == 3
    ), (
        "effective_top_n 计算错误，"
        "min(20, 3) 必须等于 3"
    )

    assert (
        dense.received_top_n
        == expected_effective_top_n
    ), (
        "Dense Retriever 收到的 top_n "
        "不是 effective_top_n："
        f"{dense.received_top_n}"
    )

    assert (
        bm25.received_top_n
        == expected_effective_top_n
    ), (
        "BM25 Retriever 收到的 top_n "
        "不是 effective_top_n："
        f"{bm25.received_top_n}"
    )

    assert (
        rrf.received_top_n
        == expected_effective_top_n
    ), (
        "RRF 收到的 top_n "
        "不是 effective_top_n："
        f"{rrf.received_top_n}"
    )

    # --------------------------------------------------------
    # 3. 反向断言
    #
    # 防止有人改回：
    #   top_n=top_n
    #
    # 一旦 top_n 被原样透传，
    # 这里会立刻失败。
    # --------------------------------------------------------

    assert (
        dense.received_top_n
        != requested_top_n
    ), (
        "Dense Retriever 收到了 "
        "未经收敛的 top_n=20"
    )

    assert (
        bm25.received_top_n
        != requested_top_n
    ), (
        "BM25 Retriever 收到了 "
        "未经收敛的 top_n=20"
    )

    assert (
        rrf.received_top_n
        != requested_top_n
    ), (
        "RRF 收到了 "
        "未经收敛的 top_n=20"
    )

    # --------------------------------------------------------
    # 4. BM25 必须拿到完整 Corpus
    #
    # bm25s 报错的前提就是：
    # k 大于 corpus size。
    # --------------------------------------------------------

    assert (
        len(bm25.received_texts)
        == corpus_size
    ), (
        "BM25 Index 没有收到完整 Corpus"
    )

    assert (
        dense.received_memory_count
        == corpus_size
    ), (
        "Dense Retriever 收到的 "
        "memories 数量不对"
    )

    # --------------------------------------------------------
    # 5. RRF 收到的 ranking 必须覆盖全部 Corpus
    # --------------------------------------------------------

    assert (
        rrf.received_rankings[0]
        == [0, 1, 2]
    ), (
        "Dense Ranking 不符合预期"
    )

    assert (
        rrf.received_rankings[1]
        == [0, 1, 2]
    ), (
        "BM25 Ranking 不符合预期"
    )

    # --------------------------------------------------------
    # 6. 最终结果必须是 3 条
    # --------------------------------------------------------

    assert (
        len(result.items)
        == corpus_size
    ), (
        "最终 items 数量不是 3"
    )

    # --------------------------------------------------------
    # 7. top_k=5 > items=3 不得出错
    #
    # items[:5] 对只有 3 条的 items 合法。
    #
    # 所以 Judge 最多处理 3 条。
    # --------------------------------------------------------

    assert (
        len(reranker.received_texts)
        == corpus_size
    ), (
        "Reranker 收到的 Candidate "
        "数量不是 3"
    )

    assert (
        len(judge.received_texts)
        == corpus_size
    ), (
        "top_k=5 但只有 3 条 Candidate 时，"
        "Judge 应该只处理 3 条，"
        f"实际 {len(judge.received_texts)} 条"
    )

    # --------------------------------------------------------
    # 8. 每条 item 必须保留完整 Runtime Trace
    # --------------------------------------------------------

    for item in result.items:

        assert (
            item.dense_score
            is not None
        ), (
            "Dense Trace 丢失"
        )

        assert (
            item.bm25_score
            is not None
        ), (
            "BM25 Trace 丢失"
        )

        assert (
            item.bm25_score > 0
        ), (
            "BM25 Score 应该为正"
        )

        assert (
            item.rrf_score > 0
        ), (
            "RRF Score 应该为正"
        )

        assert (
            item.rerank_score
            is not None
        ), (
            "Rerank Trace 丢失"
        )

        assert (
            item.judge_selected
            is True
        ), (
            "Fake Judge 应该接受所有 Candidate"
        )

    # --------------------------------------------------------
    # 9. Injector 最终收到 3 条
    # --------------------------------------------------------

    assert (
        len(injector.received_texts)
        == corpus_size
    ), (
        "Injector 收到的文本数量不是 3"
    )

    assert (
        result.memory_context
    ), (
        "memory_context 不应为空"
    )

    print()
    print(
        "[PASS] CASE 5 - "
        "TOP_N LARGER THAN CORPUS"
    )


# ============================================================
# Main
# ============================================================


def main():

    print()
    print("=" * 80)
    print("MEMORY READ V2 - REAL E2E TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # SQLite 使用 sqlite:///./test.db
    #
    # ./ 是相对于当前工作目录。
    #
    # 所以先明确打印实际 DB。
    # --------------------------------------------------------

    project_root = Path.cwd()

    database_path = (
        project_root / "test.db"
    )

    print()
    print(
        f"Current Working Directory: "
        f"{project_root}"
    )

    print(
        f"SQLite Database: "
        f"{database_path.resolve()}"
    )

    # --------------------------------------------------------
    # 防止从错误目录执行测试，
    # 导致 SQLite 自动连接到另一个 test.db。
    # --------------------------------------------------------

    if not database_path.exists():
        raise RuntimeError(
            "当前目录下找不到 test.db。\n"
            "请从项目根目录运行：\n"
            "F:\\AI_Agent_Platform"
        )

    # --------------------------------------------------------
    # Case 5
    #
    # Controlled Boundary Test。
    #
    # 先执行，
    # 因为它不依赖任何外部资源：
    #
    # SQLite / Embedding / BM25 /
    # Reranker / DeepSeek
    #
    # 即使外部资源不可用，
    # 这条 Regression Test 的结果
    # 也一定可见。
    # --------------------------------------------------------

    test_top_n_larger_than_corpus()

    # --------------------------------------------------------
    # Real Components
    #
    # 不注入 Fake：
    #
    # Repository   -> Real SQLite
    # Dense        -> Real Qwen
    # BM25         -> Real bm25s / jieba
    # RRF          -> Real
    # Reranker     -> Real bge
    # Judge        -> Real DeepSeek
    # Injector     -> Real
    # --------------------------------------------------------

    reader = MemoryReader()

    db = SessionLocal()

    try:

        test_langgraph_lexical(
            reader=reader,
            db=db
        )

        test_langgraph_semantic(
            reader=reader,
            db=db
        )

        test_no_relevant_memory(
            reader=reader,
            db=db
        )

        test_no_memory(
            reader=reader,
            db=db
        )

    finally:

        db.close()

        print()
        print("[DB] Session closed")

    print()
    print("=" * 80)
    print(
        "ALL MEMORY READ V2 "
        "REAL E2E TESTS PASSED"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()