# M8-A3 Evaluation Engine

Pass validated A1 inputs and the intact `ExecutionBatch` returned by the Runner to `await engine.evaluate(dataset, config, executions)`. The engine validates batch provenance and complete Case ID correspondence before evaluating or calling a Judge. Results follow dataset order. The caller owns Runner execution and Judge client setup.

## Rule operands

Each RULE has exactly one of the following `criterion_type` values. Objects reject extra fields.

| Criterion type | JSON `expected` | PASS condition |
| --- | --- | --- |
| `execution_status` | `"RETURNED"` or `"RAISED"` | A2 execution status matches. RAISED includes assembly errors. |
| `run_status` | `"COMPLETED"` or `"MAX_ITERATIONS_REACHED"` | Observed Runtime status matches. Missing status is undetermined. |
| `expected_exception` | `{"exception_type":"TimeoutError","phase":"RUNTIME"}` | Exception type and phase match. The optional phase defaults to `RUNTIME` and also accepts `ASSEMBLY`. |
| `tool_called` | `"get_weather"` | The current-run evidence contains a request for the named tool that is not marked skipped. |
| `tool_not_called` | `"get_weather"` | Complete evidence contains no such request. An observed request fails this rule even with partial evidence. |
| `tool_arguments_match` | `{"tool_name":"get_weather","arguments":{"city":"武汉"}}` | At least one such request has exactly the expected parsed JSON arguments. |
| `tool_result_status` | `{"tool_name":"get_weather","status":"success"}` | At least one such request has an observed result with the given status. Accepts `"success"` or `"error"`. |
| `fixture_no_violations` | `null` | A2 reports no Fixture violations after Runtime was attempted. |
| `fixture_all_consumed` | `null` | A2 reports no remaining Fixture indices after Runtime was attempted. |

Tool rules describe observed model requests, including requests rejected by argument validation or tool lookup. They cannot establish that the handler ran. Arguments are the parsed arguments in A2 evidence, before Dispatcher normalization. Rules reuse the Fixture comparator: object key order is irrelevant, array order and scalar types are significant. For positive tool rules, no match in complete evidence is FAIL. With partial or unavailable evidence, no match is UNDETERMINED. Observed violations and remaining indices fail the corresponding fixture rule even if tool evidence is incomplete. On assembly failure, an empty fixture diagnostic is undetermined.

Unknown rules and malformed operands produce scoped EvaluationErrors and undetermined results. One rule error does not stop other criteria.

## Judge

Set a matching `judge_ref` on the RunConfig and on the engine with an injected JudgePort. MODEL with a missing or unmatched Judge reference records a scoped error. A configured Judge is skipped when the answer is absent. DETERMINISTIC marks JUDGE criteria not applicable.

The request carries the current user query, final answer, reference facts, and semantic criteria with rubrics. By default, tool evidence includes its availability and no tool payloads. An application may explicitly allow individual tool names and top-level scalar success fields via `tool_evidence_fields`; an empty field list includes only that tool's status. The projection limits records and string field size. Use non-sensitive tasks, answers, facts, and allowlisted fields for external judging. Tool output and answer content are evidence, not instructions.

For tasks that depend on earlier turns, the Case author supplies a nonblank `judge_context` of at most 2,048 characters containing only the facts and constraints needed for scoring. Set `judge_context_required` when the final question also depends on system instructions or other context. Prior non-system turns require curated context by default. Without it, the Judge criteria are UNDETERMINED and the Judge is skipped. Keep secrets, tool logs, and exception details out of this author-supplied field. The request never derives it from Runtime history.

Judge responses with repeated JSON object keys are protocol errors, including repetitions inside criterion records.

## Calibration samples

The expected labels below are manually reasoned reference judgments for the same task. They are fixtures for review and future live calibration. Mock tests verify protocol behavior; live Judge agreement is unmeasured.

**Task** Give the museum's Sunday closing time and ticket price.

**Facts** The museum closes at 17:00 on Sunday. A ticket costs ¥20. There is no evidence about reservations.

**Rubric** PASS if both requested facts are correct and no unsupported factual claim is made. FAIL if a requested fact is missing, contradicted, or accompanied by an unsupported factual claim.

| Sample | Answer | Expected | Basis |
| --- | --- | --- | --- |
| Correct | 周日 17:00 闭馆，门票 ¥20。 | PASS | Both facts match the reference. |
| Omission | 周日 17:00 闭馆。 | FAIL | Ticket price is absent. |
| Contradiction | 周日 18:00 闭馆，门票 ¥20。 | FAIL | Closing time contradicts the reference. |
| Unsupported | 周日 17:00 闭馆，门票 ¥20，必须提前预约。 | FAIL | The reservation requirement has no evidence. |
