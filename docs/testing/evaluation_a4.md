# M8-A4 regression harness

## Acceptance status

| Scope | Status |
| --- | --- |
| Module 8 Part A Evaluation V1 | COMPLETE / FROZEN (Offline V1) |
| Real Judge Calibration | PENDING |
| Module 8 Part B Observability | NEXT |
| Stage 3 | IN PROGRESS |

## Offline entry point

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m backend.evaluation --output .pytest_cache/m8a4/report.json
```

The CLI loads the bundled A1 dataset and run config. It selects the versioned FakeLLM script profile, runs the A2 Runner with the production tool schemas and fixture handlers, evaluates the A2 records with the A3 Engine, then computes metrics and writes the JSON projection. Use `--dataset` and `--config` to select alternate inputs compatible with the bundled script profile. The CLI exit code indicates whether the report was produced; use the report verdicts for regression assertions. The bundled run intentionally includes FAIL and UNDETERMINED cases.

The CLI rejects MODEL. For an explicitly configured MODEL run, call `await run_evaluation(dataset, config, runner=runner, engine=engine, model_case_limit=n)` with injected model and Judge ports and a matching profile configuration. MODEL requires a dataset declared as `AGENT_QUALITY`, with a mandatory semantic Judge criterion in every Case. The Case author must define a rubric covering the task's substantive requirements. Its adequacy still requires review. The case limit must cover the batch and is capped at 20. Engine makes at most one Judge invocation per eligible case. The supplied DeepSeek Judge Adapter disables SDK retries and bounds output tokens and request time. These constraints cap adapter requests without estimating money. Arbitrary injected ports are responsible for their own downstream calls.

## Dataset coverage

The V1 dataset has 15 cases and is explicitly marked `HARNESS_REGRESSION`. Datasets without a purpose also default to that classification. The iteration limit and expected Runtime exception cases are behavior probes for the harness. This dataset runs in DETERMINISTIC mode. Its synthetic weather fixtures use the production `get_weather` name and city schema. Tool results use the production payload shape. Responses live in a separate FakeLLM profile. Fixture values and museum facts are fixed test inputs. Fixture arguments are prevalidated and normalized with the same registered schema and serialization settings used for actual calls. Invalid parameters abort the batch before any Agent executes.

| Scenarios | Coverage |
| --- | --- |
| `no_tool`, `correct_tool`, `wrong_tool`, `wrong_arguments` | Direct response, correct dispatch, unknown tool, invalid city arguments. |
| `multiple_calls`, `recover_after_tool_failure` | Ordered lookups and failure followed by recovery. |
| `successful_tool_wrong_answer`, `tool_failure_task_failure` | Separate tool success from answer correctness and exercise unrecovered tool failure. |
| `fixture_mismatch`, `fixture_exhausted` | Distinct fixture violations and unused fixture counts. |
| `max_iterations`, `expected_runtime_exception` | Partial tool evidence at the iteration limit and a passing expected exception. |
| `reference_contradiction`, `correct_without_tool_evidence`, `judge_uncertainty` | Conflicting facts, a correct value without a required lookup, and an unverified reservation claim. |

## Metrics and projection

Each report represents one Runner-issued batch. Its grouping key uses the dataset fingerprint, mode, run config fingerprint and, for DETERMINISTIC runs, the script content fingerprint. The script fingerprint covers selected Case IDs in dataset order and each selected Case's ordered response sequence, including tool call details and responses unused because the Agent stopped early. Missing scripts have a distinct marker. Entries for Case IDs outside the batch do not contribute. Raw scripts are excluded from the report. Bind model and Judge profile references to versioned dependencies.

Keep the `ExecutionBatch` returned by `EvaluationRunner.run()` intact for Engine and Report. It supports indexing and iteration. Passing a plain list or replacing records requires rerunning through the Runner. Batch provenance guards against incorrect input pairing in the current process. It is not a cross-process signature or a security boundary against code with access to Python internals.

- Confirmed pass rate is PASS cases divided by all cases, including UNDETERMINED.
- Runtime counters separate RETURNED and RAISED, and split returned results into COMPLETED and MAX_ITERATIONS_REACHED.
- Rule and Judge decision counts each use all criteria of that evaluator type, including NOT_APPLICABLE. Evaluation errors are counted both by affected case and by individual error record. No tool accuracy rate is inferred.
- Tokens, cost, and model latency are `null` because these values are unavailable. Case duration is omitted so the bundled regression report is byte-reproducible.

The report validates the complete Case and Criterion correspondence against the Dataset, including required IDs and evaluator types. An error explicitly scoped to a known optional result is recorded without vetoing required success. A global error, an error on a required item, or a missing required decision blocks PASS. A confirmed required FAIL takes precedence.

Case and criterion indices are zero-based positions in the fingerprinted dataset. The export includes verdicts, fixed reason codes, allowlisted evaluation error codes, runtime statuses, evidence availability, and fixture diagnostic counts. Unknown error codes are mapped to a fixed category. The projection excludes raw IDs, history, answers, prompts, arguments, tool payloads, Judge rationales, exception data and reference facts. Dataset and profile fingerprints identify a run configuration. They are not a substitute for keeping credentials out of evaluation inputs.

## Judge acceptance

The four A3 calibration samples use an explicit two-fact task with fixed fictional ground truth. Under that rubric, the complete answer passes, the missing price and conflicting time fail, and the unsupported reservation claim fails because the rubric explicitly penalizes added claims. A separate A4 boundary case asks only about reservations. The supplied facts do not establish that policy, so the expected semantic decision is UNDETERMINED. The reference labels are reasoned from the supplied fixtures and are ready for human sign-off.

Offline acceptance after the final review fixes passed. The bundled CLI produced 6 PASS, 6 FAIL, and 3 UNDETERMINED across 15 scripted cases. Evaluation passed 209 tests, Runtime passed 372 tests, and the combined run passed 581 tests. Offline Judge protocol checks use FakeJudge and mocked SDK transport.

Real Judge API compatibility and live calibration have not been verified. Before treating MODEL scores as quality evidence, review the reference labels and explicitly authorize a bounded live run with pinned Agent and Judge profiles.
