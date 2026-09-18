"""
MemoryRelationshipJudge Structured Output Contract
[KEEP] Deterministic Tests
===========================================================

背景：

MemoryRelationshipJudge 已经由

    call_llm()
    → str
    → json.loads()
    → 手工解析与逐条校验

迁移为

    call_llm_structured(
        messages=...,
        output_model=MemoryRelationshipLLMOutput,
    )
    → MemoryRelationshipLLMOutput
    → Runtime Business Validation
    → MemoryRelationshipResult

迁移之前，本组件的 LLM Boundary 层
**没有任何 deterministic 测试**：

    test_memory_lifecycle_v1.py 里
    只有 FakeRelationshipJudge（fake 整个 Judge）
    与 validate_relationship_coverage() 的直接测试。

因此本文件的首要目的不是"让测试变绿"，
而是先补一层可靠的安全网，
再证明 Structured Output Migration
没有破坏旧 Business Contract。

本文件覆盖的 Contract
===========================================================

Part A — Judge LLM Boundary / Business Conversion

    1.  existing_memories=[] → [] 且不调用 LLM
    2.  合法单条 Relationship → MemoryRelationship
    3.  合法多条 Relationship → 全部保留
    4.  output_model is MemoryRelationshipLLMOutput
    5.  Per-item Skip Contract（逐条过滤，同批次互不牵连）
    6.  Runtime memory_id Validation（动态 valid_memory_ids）
    7.  relationships=None → []
    8.  relationships 非 list → Structured Validation Failure
    9.  Pydantic ValidationError → []
    10. Provider RuntimeError → []（fail-closed）
    11. item 三形态：dict / MemoryRelationshipLLMItem / garbage
    12-15. System Prompt 业务语义

Part B — Coverage Contract

    validate_relationship_coverage() 的职责，
    与 Judge 的逐条过滤**严格分离**：

        Judge       允许返回部分 relationships
        Coverage    要求完整覆盖，否则
                    raise RelationshipContractError

    本文件只补 test_memory_lifecycle_v1.py
    尚未覆盖的两项（非法 relationship / 空白 reason），
    不重复其已有的 missing / duplicate / unknown 用例。

设计约束：

    deterministic
        不联网
        不调用真实 DeepSeek
        不加载真实 Embedding
        不加载真实 Reranker
        不写真实 test.db / backend.db

    LLM 通过 patch 模块级 call_llm_structured 注入 Fake。

Async Contract 注意事项：

    MemoryRelationshipJudge.judge 已经是 async Contract。

    本模块的 test_ 函数全部保持同步 def test_xxx()，
    由 run() 把 coroutine 驱动到底：

        1. 直接 python 运行时会真的执行
        2. pytest 会原生收集执行，
           不会被当成 async test 静默跳过
        3. coroutine 一定被 await
        4. 不会产生假 PASS

    本模块不依赖 pytest-asyncio。

运行：

    .venv/Scripts/python.exe tests/memory/test_memory_relationship_judge_structured_output.py
"""


import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


# ============================================================
# Project Root Bootstrap
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from pydantic import ValidationError  # noqa: E402

from backend.schemas.memory_candidate import (  # noqa: E402
    MemoryCandidate,
)

from backend.schemas.memory_relationship import (  # noqa: E402
    MemoryRelationship,
    MemoryRelationshipResult,
)

from backend.memory.memory_write.memory_relationship_judge import (  # noqa: E402
    ALLOWED_RELATIONSHIPS,
    MemoryRelationshipJudge,
    MemoryRelationshipLLMItem,
    MemoryRelationshipLLMOutput,
    RelationshipContractError,
    validate_relationship_coverage,
)


# ============================================================
# Patch Target
#
# 生产模块在模块级直接 import call_llm_structured，
# 因此 patch 该模块自己的引用。
# ============================================================

JUDGE_CALL_LLM_STRUCTURED = (
    "backend.memory.memory_write."
    "memory_relationship_judge.call_llm_structured"
)


# ============================================================
# Helpers
# ============================================================


def run(coro):
    """
    在同步测试中真实执行并等待
    async production contract。

    只负责把 coroutine 驱动到底，
    不参与任何业务断言。
    """

    return asyncio.run(coro)


class FakeRelationshipStructuredLLM:
    """
    Fake call_llm_structured。

    生产 Contract：

        await call_llm_structured(
            messages=messages,
            output_model=MemoryRelationshipLLMOutput,
        )
        → MemoryRelationshipLLMOutput

    因此本 Fake 必须保持同样的
    keyword-only Calling Contract，
    并返回 Structured Output Model，
    而不是旧实现里的 JSON str。
    """

    def __init__(
        self,
        output=None,
        exc=None
    ):
        self._output = output
        self._exc = exc

        self.call_count = 0
        self.received_messages = None
        self.received_output_model = None
        self.received_model = None

    async def __call__(
        self,
        *,
        messages,
        output_model,
        model=None
    ):
        self.call_count += 1

        self.received_messages = messages
        self.received_output_model = output_model
        self.received_model = model

        assert (
            output_model
            is MemoryRelationshipLLMOutput
        ), (
            "MemoryRelationshipJudge 必须以 "
            "MemoryRelationshipLLMOutput "
            "作为 output_model，"
            f"实际为：{output_model}"
        )

        if self._exc is not None:

            raise self._exc

        if self._output is None:

            raise AssertionError(
                "FakeRelationshipStructuredLLM "
                "未配置输出"
            )

        return self._output


def rel(
    memory_id,
    relationship="new",
    reason="fake reason"
):
    """
    构造一条 relationships dict。

    默认使用 dict 形态：
    真实 Provider 路径就是从 JSON 解析出 dict，
    再交给 Pydantic 校验，
    因此 dict 是对真实输入最忠实的还原。

    MemoryRelationshipLLMItem 形态
    由 test_llm_output_accepts_dict_and_model_items_
    and_skips_garbage 单独覆盖。
    """

    return {
        "memory_id": memory_id,
        "relationship": relationship,
        "reason": reason,
    }


def relationship_output(*relationships):
    """
    构造 MemoryRelationshipLLMOutput。
    """

    return MemoryRelationshipLLMOutput(
        relationships=list(relationships)
    )


@dataclass
class ExistingMemory:
    """
    Existing Memory 的最小输入形态。

    Judge 只读取：
        memory_id
        content
        memory_type
        similarity
    """

    memory_id: int

    content: str = "existing content"

    memory_type: str = "fact"

    similarity: float = 0.9


def existing(memory_id):
    return ExistingMemory(
        memory_id=memory_id
    )


def make_candidate():
    return MemoryCandidate(
        content="用户喜欢 Python",
        memory_type="preference",
    )


def run_judge(
    existing_memories,
    output=None,
    exc=None,
):
    """
    用 Fake call_llm_structured 运行真实
    MemoryRelationshipJudge。

    返回：

        (result, fake_llm)
    """

    fake_llm = FakeRelationshipStructuredLLM(
        output=output,
        exc=exc,
    )

    judge = MemoryRelationshipJudge()

    with patch(
        JUDGE_CALL_LLM_STRUCTURED,
        fake_llm,
    ):

        result = run(
            judge.judge(
                make_candidate(),
                existing_memories,
            )
        )

    return result, fake_llm


def ids_of(result):
    return [
        item.memory_id
        for item in result.relationships
    ]


def system_prompt_of(messages):
    return messages[0]["content"]


def assert_prompt_contains(
    prompt,
    tokens,
    label
):
    missing = [
        token
        for token in tokens
        if token not in prompt
    ]

    assert not missing, (
        f"{label}：System Prompt 缺少关键语义 "
        f"{missing}"
    )


def build_prompt_text():
    """
    跑一次真实 Judge，取出 System Prompt。
    """

    _, fake_llm = run_judge(
        existing_memories=[existing(1)],
        output=relationship_output(
            rel(1, "new")
        ),
    )

    return system_prompt_of(
        fake_llm.received_messages
    )


# ============================================================
# ============================================================
# Part A — Judge LLM Boundary / Business Conversion
# ============================================================
# ============================================================


def test_no_existing_memories_returns_empty_without_llm():
    """
    Contract：

    existing_memories=[]
        ↓
    MemoryRelationshipResult(relationships=[])
        ↓
    **不调用 LLM**

    没有 Existing Memory 时，
    Relationship 判断在语义上不存在，
    因此必须短路，不允许产生任何 LLM 调用。
    """

    result, fake_llm = run_judge(
        existing_memories=[],
        output=relationship_output(
            rel(1, "new")
        ),
    )

    assert isinstance(
        result,
        MemoryRelationshipResult
    )

    assert result.relationships == []

    assert fake_llm.call_count == 0, (
        "没有 Existing Memory 时不应调用 LLM"
    )


def test_single_valid_relationship_maps_to_business_result():
    """
    Contract：

    Structured Output 中一条合法 Relationship
        ↓
    MemoryRelationship

    并且 reason 会被 strip。

    （Judge 是唯一对 reason 做 strip 的地方，
      旧实现同样如此。）
    """

    result, _ = run_judge(
        existing_memories=[existing(1)],
        output=relationship_output(
            rel(
                1,
                "duplicate",
                "  两条 Memory 表达同一偏好  ",
            )
        ),
    )

    assert len(result.relationships) == 1

    item = result.relationships[0]

    assert isinstance(
        item,
        MemoryRelationship
    ), "必须转换为业务层 MemoryRelationship"

    assert item.memory_id == 1

    assert item.relationship == "duplicate"

    assert item.reason == (
        "两条 Memory 表达同一偏好"
    ), (
        "reason 必须去除首尾空白，"
        f"实际：{item.reason!r}"
    )


def test_multiple_valid_relationships_are_all_kept():
    """
    Contract：

    多个合法 Relationship
        ↓
    全部保留，且保持输出顺序。
    """

    result, _ = run_judge(
        existing_memories=[
            existing(1),
            existing(2),
            existing(3),
        ],
        output=relationship_output(
            rel(1, "duplicate"),
            rel(2, "conflict"),
            rel(3, "new"),
        ),
    )

    assert [
        (item.memory_id, item.relationship)
        for item in result.relationships
    ] == [
        (1, "duplicate"),
        (2, "conflict"),
        (3, "new"),
    ]


def test_judge_requests_relationship_llm_output_model():
    """
    Contract：

    Judge 必须向 Provider 声明
    自己需要的是 MemoryRelationshipLLMOutput。

    这是 Structured Output 迁移之后
    新增的、真实的 Judge 责任：
    声明输出 Schema。
    """

    _, fake_llm = run_judge(
        existing_memories=[existing(1)],
        output=relationship_output(
            rel(1, "new")
        ),
    )

    assert (
        fake_llm.received_output_model
        is MemoryRelationshipLLMOutput
    ), (
        "Judge 必须以 "
        "MemoryRelationshipLLMOutput "
        "作为 output_model"
    )


def test_per_item_skip_contract():
    """
    ★ 核心 Business Contract ★

    一条坏 Relationship
    不得拖死同批次其他合法 Relationship。

    Judge **允许**返回部分 relationships
    （完整覆盖由 validate_relationship_coverage
      负责，见 Part B）。

    输入 A/B/C 时：

        A 合法
        B 非法
        C 合法
            ↓
        [A, C]
            ↓
        不能是 []

    非法形态包括：

        relationship 不在白名单
        memory_id 非 int
        memory_id 未知
        reason 非 str
        reason 空串 / 纯空白
        非 dict garbage item

    这里用一张表覆盖，
    每行同时断言「坏条目被跳过」+「好条目被保留」。

    故意**不**把每种非法形态拆成
    多个名字不同、实际走同一路径的测试：
    逐条 skip 是**同一条**代码路径，
    拆开只会让测试数量虚增而非覆盖变广。

    所有用例的 existing IDs 都是 {1, 2, 3}。
    """

    cases = (
        (
            "全部合法",
            [
                rel(1, "duplicate"),
                rel(2, "conflict"),
                rel(3, "new"),
            ],
            [1, 2, 3],
        ),
        (
            "★ A 合法 / B 非法 relationship / C 合法",
            [
                rel(1, "new"),
                rel(2, "banana"),
                rel(3, "new"),
            ],
            [1, 3],
        ),
        (
            "★ A 非法 relationship / B 合法",
            [
                rel(1, "banana"),
                rel(2, "new"),
            ],
            [2],
        ),
        (
            "★ A 合法 / B 未知 memory_id / C 合法",
            [
                rel(1, "new"),
                rel(999, "new"),
                rel(3, "new"),
            ],
            [1, 3],
        ),
        (
            "memory_id 非 int",
            [
                rel("1", "new"),
                rel(2, "new"),
            ],
            [2],
        ),
        (
            "reason 非 str",
            [
                rel(1, "new", 123),
                rel(2, "new"),
            ],
            [2],
        ),
        (
            "reason 空串",
            [
                rel(1, "new", ""),
                rel(2, "new"),
            ],
            [2],
        ),
        (
            "reason 纯空白",
            [
                rel(1, "new", "   \n\t "),
                rel(2, "new"),
            ],
            [2],
        ),
        (
            "非 dict garbage item",
            [
                rel(1, "new"),
                "garbage",
                rel(3, "new"),
            ],
            [1, 3],
        ),
    )

    for (
        title,
        relationships,
        expected_ids,
    ) in cases:

        result, _ = run_judge(
            existing_memories=[
                existing(1),
                existing(2),
                existing(3),
            ],
            output=relationship_output(
                *relationships
            ),
        )

        assert ids_of(result) == expected_ids, (
            f"{title}："
            "非法条目必须被跳过，"
            "合法条目必须全部保留，"
            f"实际为 {ids_of(result)}"
        )


def test_runtime_memory_id_validation():
    """
    ★ Runtime Business Validation ★

    memory_id 必须属于本次调用根据
    existing_memories 动态构造的 valid_memory_ids。

    这**不能**只靠 Pydantic Schema 表达：

        Schema 只知道字段类型，
        不可能知道本次运行时
        究竟存在哪些 memory_id。

    用例（使用任务指定的形态）：

        existing IDs = {10, 20}

        Structured Output：
            10   → 合法
            999  → 非法（未知）
            20   → 合法

        Judge 返回：
            [10, 20]

    并且额外验证 valid_memory_ids
    是**每次调用重新构造**的，
    不是模块级常量：

        换一组 existing 再跑一次，
        同一批 LLM 输出必须得到不同结果。
    """

    # --------------------------------------------------
    # 第一轮：existing = {10, 20}
    # --------------------------------------------------

    result, _ = run_judge(
        existing_memories=[
            existing(10),
            existing(20),
        ],
        output=relationship_output(
            rel(10, "conflict"),
            rel(999, "new"),
            rel(20, "related"),
        ),
    )

    assert ids_of(result) == [10, 20], (
        "只有本次 existing 中存在的 memory_id "
        "才能通过 Runtime Validation，"
        f"实际为 {ids_of(result)}"
    )

    assert [
        item.relationship
        for item in result.relationships
    ] == ["conflict", "related"], (
        "合法条目的 relationship 必须原样保留"
    )

    # --------------------------------------------------
    # 第二轮：existing = {7}
    #
    # 同一份 LLM 输出（10 / 999 / 20）
    # 在新集合下应当全部被跳过。
    # --------------------------------------------------

    result2, _ = run_judge(
        existing_memories=[
            existing(7),
        ],
        output=relationship_output(
            rel(10, "conflict"),
            rel(999, "new"),
            rel(20, "related"),
        ),
    )

    assert result2.relationships == [], (
        "valid_memory_ids 必须按每次调用重新构造；"
        "10 / 999 / 20 都不在 {7} 中，"
        "应当全部被跳过，"
        f"实际为 {ids_of(result2)}"
    )


def test_relationships_none_returns_empty():
    """
    Contract：

    relationships=None
        ↓
    []

    顶层 before validator 把 None
    归一为 []，
    因此 Judge 走正常路径返回空结果，
    **不是**异常路径。

    这与旧实现一致：
        data.get("relationships", [])
        → None（key 存在）
        → not isinstance(None, list)
        → []
    """

    result, _ = run_judge(
        existing_memories=[existing(1)],
        output=MemoryRelationshipLLMOutput(
            relationships=None
        ),
    )

    assert result.relationships == []


def test_llm_output_rejects_non_list_relationships():
    """
    Model Boundary Contract：

    relationships 必须是 list。

        relationships="bad"

    不能静默变成 []，
    必须直接抛 ValidationError
    （由顶层 before validator raise ValueError 触发）。

    注意这里测的是 **Model 边界**；
    该 ValidationError 落到 Judge 上时
    会走 fail-closed，
    由 test_pydantic_validation_error_fails_closed 覆盖。
    """

    try:

        MemoryRelationshipLLMOutput(
            relationships="bad"
        )

    except ValidationError:

        return

    raise AssertionError(
        "relationships 不是 list 时，"
        "MemoryRelationshipLLMOutput 必须拒绝"
    )


def build_real_validation_error():
    """
    真实构造一个 ValidationError：

        MemoryRelationshipLLMOutput(
            relationships="bad"
        )

    这正是 Provider 返回无法通过
    Structured Output 校验的内容时，
    向上抛出的异常类型。
    """

    try:

        MemoryRelationshipLLMOutput(
            relationships="bad"
        )

    except ValidationError as exc:

        return exc

    raise AssertionError(
        "relationships 不是 list，"
        "MemoryRelationshipLLMOutput 应该拒绝"
    )


def test_pydantic_validation_error_fails_closed():
    """
    Contract：

    Pydantic / Structured Output Validation Error
        ↓
    MemoryRelationshipResult(relationships=[])
        ↓
    不向上传播

    旧实现同样如此：

        except json.JSONDecodeError → []
        except Exception → []

    迁移后统一由

        except Exception → []

    收敛。

    这里用**真实构造**的 ValidationError，
    而不是随手捏一个异常类型。
    """

    error = build_real_validation_error()

    result, fake_llm = run_judge(
        existing_memories=[existing(1)],
        exc=error,
    )

    assert isinstance(
        result,
        MemoryRelationshipResult
    )

    assert result.relationships == [], (
        "Structured Validation Error "
        "必须 fail-closed 返回 []"
    )

    assert fake_llm.call_count == 1, (
        "失败路径也必须真实调用过一次 "
        "call_llm_structured"
    )


def test_provider_runtime_error_fails_closed():
    """
    Contract：

    Provider / 网络层 RuntimeError
        ↓
    MemoryRelationshipResult(relationships=[])
        ↓
    **不向上传播**

    这是本组件与 Extractor 的关键差异：

        MemoryExtractor      except (ValidationError, ValueError)
                             → RuntimeError 向上传播

        MemoryRelationship   except Exception
        Judge                → RuntimeError 也收敛为 []

    之所以可以接受：
    Judge 返回 [] 之后，
    validate_relationship_coverage() 会发现
    「输出数量 != 输入数量」并 raise
    RelationshipContractError，
    由 Pipeline Fail Closed（跳过本 Candidate）。

    因此这里同时验证**两层**：

        1. Judge 返回 []
        2. 该结果确实会被 Coverage 判为 Fail Closed
    """

    result, fake_llm = run_judge(
        existing_memories=[
            existing(1),
            existing(2),
        ],
        exc=RuntimeError(
            "fake-provider-down"
        ),
    )

    assert result.relationships == [], (
        "Provider RuntimeError "
        "必须 fail-closed 返回 []"
    )

    assert fake_llm.call_count == 1

    # --------------------------------------------------
    # 关键后续：空结果必须被 Coverage Fail Closed
    #
    # 否则 Provider 故障会被伪装成
    # "Candidate 与所有 Memory 都是 new"，
    # 进而错误地写入数据库。
    # --------------------------------------------------

    try:

        validate_relationship_coverage(
            [
                existing(1),
                existing(2),
            ],
            result,
        )

    except RelationshipContractError:

        pass

    else:

        raise AssertionError(
            "Judge 返回 [] 时，"
            "Coverage 必须 raise "
            "RelationshipContractError"
        )


def test_llm_output_accepts_dict_and_model_items_and_skips_garbage():
    """
    Model Boundary Contract：

        dict                      → 正常
        MemoryRelationshipLLMItem → 正常
        其他（如 "garbage"）        → skip

    注意不要复现 Extractor 曾经出现过的
    Model Instance Trap：

    早期 MemoryExtractorLLMOutput 的
    before validator 只保留 isinstance(item, dict)，
    导致传 Model 实例被**静默过滤**成空列表。

    MemoryRelationshipLLMOutput 当前保留

        isinstance(item, (dict, MemoryRelationshipLLMItem))

    这里显式固定该契约，
    并且做一次端到端验证
    （Model 实例必须能穿过真实 Judge）。
    """

    # --------------------------------------------------
    # 1. 两种形态都能被 Model 接受
    # --------------------------------------------------

    as_dict = MemoryRelationshipLLMOutput(
        relationships=[rel(1, "new")]
    )

    assert len(as_dict.relationships) == 1

    as_model = MemoryRelationshipLLMOutput(
        relationships=[
            MemoryRelationshipLLMItem(
                memory_id=1,
                relationship="new",
                reason="fake reason",
            )
        ]
    )

    assert len(as_model.relationships) == 1, (
        "MemoryRelationshipLLMItem 形态必须被接受，"
        f"实际 {len(as_model.relationships)} 条"
    )

    # --------------------------------------------------
    # 2. 混排 + garbage
    # --------------------------------------------------

    mixed = MemoryRelationshipLLMOutput(
        relationships=[
            rel(1, "new"),
            "garbage",
            123,
            MemoryRelationshipLLMItem(
                memory_id=3,
                relationship="related",
                reason="fake reason",
            ),
        ]
    )

    assert [
        item.memory_id
        for item in mixed.relationships
    ] == [1, 3], (
        "只有 dict 与 MemoryRelationshipLLMItem "
        "应当被保留，"
        "非 dict garbage 必须被 skip"
    )

    # --------------------------------------------------
    # 3. 端到端：Model 实例形态必须穿过真实 Judge
    # --------------------------------------------------

    result, _ = run_judge(
        existing_memories=[
            existing(1),
            existing(3),
        ],
        output=relationship_output(
            MemoryRelationshipLLMItem(
                memory_id=1,
                relationship="new",
                reason="第一条",
            ),
            "garbage",
            MemoryRelationshipLLMItem(
                memory_id=3,
                relationship="related",
                reason="第二条",
            ),
        ),
    )

    assert ids_of(result) == [1, 3], (
        "MemoryRelationshipLLMItem 形态必须能穿过"
        "真实 Judge，"
        "非 dict garbage 仍应被跳过，"
        f"实际为 {ids_of(result)}"
    )


# ============================================================
# Prompt Business Semantics
#
# 只做关键语义包含校验，
# 不 snapshot 整段 Prompt。
#
# 目的：允许后续润色措辞，
# 但不允许把关键 Contract 删掉。
# ============================================================


def test_prompt_defines_four_relationship_kinds():
    """
    Prompt 必须明确四种 Relationship，
    并声明只能使用这四种。
    """

    prompt = build_prompt_text()

    assert_prompt_contains(
        prompt,
        [
            "duplicate",
            "conflict",
            "related",
            "new",
            "只能使用以下四种",
        ],
        "Relationship 种类",
    )

    assert ALLOWED_RELATIONSHIPS == {
        "duplicate",
        "conflict",
        "related",
        "new",
    }, (
        "ALLOWED_RELATIONSHIPS 必须与 Prompt "
        "声明的四种保持一致"
    )


def test_prompt_declares_duplicate_and_conflict_type_rules():
    """
    Prompt 必须写清楚 memory_type 的差异规则：

        duplicate 原则上要求 memory_type 相同
        conflict  不要求 memory_type 相同

    这两条直接决定 Lifecycle 的
    historical 迁移行为，
    属于关键语义，不能被润色掉。
    """

    prompt = build_prompt_text()

    assert_prompt_contains(
        prompt,
        [
            "duplicate 原则上要求两条 Memory 的",
            "memory_type 相同",
            "conflict 不要求两个 Memory 的",
            "memory_type 相同",
        ],
        "memory_type 规则",
    )


def test_prompt_declares_similarity_is_not_duplicate_proof():
    """
    Prompt 必须声明：

        Similarity 高不代表 duplicate

    否则 Reranker / Similarity 的分数
    会被 LLM 当成 duplicate 证据，
    直接导致误判为重复而丢弃 Candidate。
    """

    prompt = build_prompt_text()

    assert_prompt_contains(
        prompt,
        [
            "不代表一定 duplicate",
        ],
        "Similarity 语义",
    )


def test_prompt_declares_coverage_and_unknown_id_contract():
    """
    Prompt 必须声明：

        1. 必须为每一个 Existing Memory
           返回一个 Relationship
        2. 不得生成输入中不存在的 memory_id

    这两条与
    validate_relationship_coverage()
    的 Fail Closed 配合，
    共同构成 Coverage Contract 的
    "要求 LLM 做对" + "事后强制校验" 两层。
    """

    prompt = build_prompt_text()

    assert_prompt_contains(
        prompt,
        [
            "必须为每一个输入的 Existing Memory",
            "返回一个 Relationship",
            "不得生成输入中不存在的 memory_id",
        ],
        "Coverage / unknown id",
    )


# ============================================================
# ============================================================
# Part B — Coverage Contract
#
# 与 Part A 的 Judge 职责**严格分离**：
#
#     Judge      允许逐条 skip，
#                因此可以返回部分 relationships
#
#     Coverage   要求完整覆盖，
#                否则 raise RelationshipContractError
#
# 已有覆盖（test_memory_lifecycle_v1.py）：
#
#     missing ID
#     duplicate ID
#     unknown ID
#     完整覆盖 → 通过
#
# 这里只补尚未覆盖的两项：
#
#     非法 relationship
#     空白 reason
#
# 不重复上述已有用例。
# ============================================================
# ============================================================


def test_coverage_rejects_illegal_relationship():
    """
    validate_relationship_coverage()：

    即使覆盖完整、数量匹配，
    只要某条 relationship 不在白名单，
    必须 raise RelationshipContractError。

    注意：
    这里**不经过** MemoryRelationshipJudge。
    Judge 自己会 skip 非法 relationship，
    因此非法 relationship 永远不会
    从 Judge 的返回值里出现；
    本用例直接构造
    MemoryRelationshipResult 来验证
    Coverage 层的独立防线。
    """

    existing_memories = [
        existing(1),
        existing(2),
    ]

    result = MemoryRelationshipResult(
        relationships=[
            MemoryRelationship(
                memory_id=1,
                relationship="new",
                reason="ok",
            ),
            MemoryRelationship(
                memory_id=2,
                relationship="banana",
                reason="ok",
            ),
        ]
    )

    try:

        validate_relationship_coverage(
            existing_memories,
            result,
        )

    except RelationshipContractError:

        return

    raise AssertionError(
        "非法 relationship 时没有抛出 "
        "RelationshipContractError"
    )


def test_coverage_rejects_blank_reason():
    """
    validate_relationship_coverage()：

    reason 为 str 但 strip 后为空，
    必须 raise RelationshipContractError。
    """

    existing_memories = [
        existing(1),
    ]

    for bad_reason in (
        "",
        "   ",
        "\n\t ",
    ):

        result = MemoryRelationshipResult(
            relationships=[
                MemoryRelationship(
                    memory_id=1,
                    relationship="new",
                    reason=bad_reason,
                )
            ]
        )

        try:

            validate_relationship_coverage(
                existing_memories,
                result,
            )

        except RelationshipContractError:

            continue

        raise AssertionError(
            f"reason={bad_reason!r} 时没有抛出 "
            "RelationshipContractError"
        )


# ============================================================
# Test Runner
#
# 与项目现有测试保持一致：
# 既可由 pytest 收集，也可直接 python 运行。
# ============================================================


def collect_tests():
    """
    收集本模块中所有 test_ 函数，
    按函数名排序，保证执行顺序确定。
    """

    current_globals = globals()

    names = [
        name
        for name in current_globals
        if name.startswith("test_")
        and callable(
            current_globals[name]
        )
    ]

    names.sort()

    return [
        (
            name,
            current_globals[name]
        )
        for name in names
    ]


def main():

    tests = collect_tests()

    passed = []
    failed = []

    print()
    print("=" * 70)
    print(
        "MEMORY RELATIONSHIP JUDGE "
        "STRUCTURED OUTPUT [KEEP]"
    )
    print("=" * 70)

    for name, func in tests:

        try:

            func()

        except AssertionError as exc:

            failed.append(
                (name, f"AssertionError: {exc}")
            )

            print(f"[FAIL] {name}")

        except Exception as exc:

            failed.append(
                (
                    name,
                    f"{type(exc).__name__}: {exc}",
                )
            )

            print(f"[ERROR] {name}")

        else:

            passed.append(name)

            print(f"[PASS] {name}")

    print()
    print("=" * 70)
    print(
        f"TOTAL: {len(passed)} passed / "
        f"{len(failed)} failed"
    )
    print("=" * 70)

    for name, reason in failed:

        print(f"  [FAIL] {name}: {reason}")

    return 1 if failed else 0


if __name__ == "__main__":

    raise SystemExit(main())
