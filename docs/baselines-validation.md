# Thinking baseline validation

## Unified boolean switch verification — 2026-10-12

The current notebook exposes one boolean, `ENABLE_EARLY_EXIT`, mapping to the
existing unified model/generation implementation. No separate model or generator
copies are maintained. `True` selects early_exit; `False` selects simulated.

- Full offline suite: **42 tests passed**, no skips, 60.875 seconds.
- Actual parameter/run cells with True and False build the appropriate CLI mode
  and distinct result directories. A subsequent test-only extension executes
  the real merge cell's path assignments for both modes and checks all five
  input directories plus merged output; both focused notebook tests passed.
- The real tiny-checkpoint CLI runs both modes with BF16/PTQ-labelled CPU test
  arms, verifies physical versus simulated loop counts, rejects cross-mode
  resume, and confirms fixed-threshold token equality. These remain explicitly
  invalid CPU research reports, not pretrained benchmark results.
- All 16 production/model/config/requirement files are byte-identical to the
  verified 8bb6c2bd7956 source bundle. Changes expose the boolean in the notebook
  and extend tests/documentation; previous execution semantics are preserved.
- Both notebook schemas and code cells validate, with no saved outputs.
- Independent review found no functional issue in switch, directory, resume or
  merge wiring. Its minor test-coverage observation was addressed by executing
  the actual merge path assignments in the focused test above.

## Physical early-exit verification — 2026-10-12

Full offline suite: **41 tests passed**, no skips, 51.314 seconds. Tests used
isolated dependencies on Windows / Python 3.13.5 / PyTorch 2.6.0 CPU /
Transformers 4.57.1. Full pretrained GPU evaluation remains unmeasured.

- New failing tests first demonstrated that the original core always executed
  four loops. After implementation, actual decoder-layer hooks establish that
  reaching the gate threshold stops attention/MLP execution at that depth.
- For changing depths 1, 4, 2, 3, compare logits and every K/V tensor against an
  independent full-compute oracle that manually fills the last token's skipped
  cache positions. Both DynamicCache and UniversalTransformerCache agree;
  historical positions are preserved exactly and all histories stay aligned.
- Conditional hazard probabilities [.25, .375, ...] cross 0.5 at depth 2.
  BF16 gates spanning hazards 0, .25, .5, .9, 1 and thresholds .1, .5, .9,
  .999, 1 are tested at loop caps 1, 3, 4. Threshold 1 always runs the cap.
- With saturated gates, first and subsequent generation decisions execute one
  loop; full-depth prompt prefix work is counted separately. Actual layer hooks
  match `loops*(P-1)+sum(executed_depths)`. Missing gate probabilities are null.
- Independent threshold invocations start fresh caches; repeated adaptive runs
  agree even after intervening fixed-depth runs. EOS, error cleanup, unsupported
  inputs/cache modes, training and conflicting exit strategies are covered.
- The real offline CLI runs BF16/PTQ-labelled CPU test arms at both thresholds,
  confirms physical depth traces and protocol mode/cache policy, resumes without
  duplicates, and rejects mode changes. All CPU reports are explicitly invalid
  for research accuracy. Notebook code passes the new mode and result prefix.
- Existing simulated generation still matches cached upstream-style generate;
  the fixed threshold-1 trajectory is unchanged. Smoke tests pass for 3 and 4
  loops (cached/uncached max logit errors 1.49e-7 and 1.04e-7).
- Both notebooks pass nbformat validation, compile, and contain no saved output.
  Compared with bundle bdf55b1c7ddc, local config/configuration, dataset selection,
  quantization, scoring, merger, inference entrypoint, requirements and original
  upstream hashes are byte-identical. Model, traced generation and reporting
  changes are deliberate and included in run identity.

Last-exit KV fill is a decoding approximation. It can change adaptive text,
accuracy and token length relative to previous simulated exit, and it does not
reduce allocated KV slots. GPU gate synchronization/copy cost, wallclock speed,
pretrained accuracy and MATH-500 runtime must be measured in Colab. Current mode
semantics supersede the historical limitations in the sections below.

An independent read-only reviewer found no Critical, Important or actionable
Minor issues in this change. The review additionally exercised an initially
empty implicit UniversalTransformerCache, changing depth 1 to 4 with all eight
slots aligned. Review did not establish full-checkpoint accuracy, CUDA behavior,
Colab speedup or the research quality of the cache-fill approximation.

## Batch workflow verification — 2026-10-12

Full offline suite: **34 tests passed**, no skips, 41.896 seconds. Dependencies
were loaded from isolated directories without changing the global Python setup.

- Five seeded slices of 100 cover all 500 IDs exactly once; batch 1 preserves
  the previous first-100 selection. Invalid offsets fail before inference.
- The real tiny-model CLI applies offset selection, records it in the manifest,
  runs both arms and thresholds, and resumes without duplicate results.
- The actual notebook parameter/run cells select batch 5 as offset 400 and an
  independent output directory. All 12 code cells compile; nbformat validation
  passes and no outputs are saved.
- The merger recomputes raw-record statistics, including jointly solved pairs,
  token-weighted depths and standard errors; stale per-batch summaries are ignored.
- Overlapping examples, incomplete batches, wrong total counts, protocol/runtime
  differences and corrupted identities are rejected before producing outputs.
- A separate end-to-end synthetic check merged **five 100-question ZIPs** through
  the real CLI into **2000 records**, four groups with 500 examples each, and two
  thresholds with 500 paired examples each. All five original manifests and
  quantization files were preserved. This is validation data, not measured accuracy.
- Compared with bundle `08e24de96896`, model/config, generation, quantization,
  scoring, reporting and dependency files are unchanged. New code only adds data
  slicing, selection metadata and completed-batch aggregation; notebook/docs guide
  the five-batch workflow and Drive persistence.

The five pretrained GPU batches have **not** been executed locally. The user still
needs to run them in Colab using the same source bundle, GPU model and environment.
An independent read-only reviewer found no important correctness issues and
confirmed that fresh Colab extraction paths do not enter the saved model config
identity under Transformers 4.57.1.

## Original baseline verification — 2026-10-11

### Verified locally

Command: `python -m unittest discover -s tests -v`

**27 tests passed**, no skips, final run 35.445 seconds. Runtime: Windows,
Python 3.13.5, PyTorch 2.6.0 CPU, Transformers 4.57.1, Accelerate 1.11.0,
PyArrow 20.0.0, Math-Verify 0.9.0. Dependencies were supplied through isolated
test directories; the global Python installation was not changed.

Tests cover:

- E2M1 tie-to-even and saturation, E4M3 scale rounding, zero/tiny blocks,
  BF16 QDQ output dtype, matrix-global scale independent of chunk size.
- Exactly the shared core MLP gate/up/down weights change; attention, exit
  gate, embedding, norms and output head remain untouched.
- Traced greedy tokens equal upstream cached `generate` at thresholds 1 and
  0.5 on tiny random models. Prefill counted once per first prediction, EOS
  counted once, length censoring distinguished from EOS at the output limit.
- A saturated BF16 exit gate cannot override the explicit final-loop reference;
  threshold 0.5 can select loop 1 while executed depth remains 4.
- GSM8K numeric/fraction/thousands-separator grading, MATH symbolic equivalence,
  strict final-answer extraction, incomplete thinking, and special-token-aware
  decoding. Malformed comma lists cannot become valid integers.
- Paired statistics on jointly solved questions, null SE for one observation,
  output overwrite refusal and incompatible-resume rejection.
- The real benchmark CLI runs both arms and both thresholds offline using a
  tiny saved checkpoint and local questions, then resumes without duplicate
  records and rejects a changed seed. Its reports are marked invalid CPU test
  data, not scientific BF16 results.
- Both notebooks compile with cleared outputs; actual upload cells accept the
  valid bundle, preserve edited sources, use fresh directories and reject
  modified payloads. The baseline parameter/run cells build valid CLI arguments.
- Source packaging includes new benchmark modules and excludes weights,
  credentials, environments and saved results. Existing inference CLI still
  executes edited local source and writes provenance correctly.

`python scripts/smoke_test.py`: **PASS**, 3 loops / 6 tiny cache slots and
4 loops / 8 slots. Cached versus uncached maximum logit differences:
1.49e-7 and 1.04e-7 respectively.

Additional real-data checks:

- Thinking model/config files match all 3 original SHA256 hashes at revision
  `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`.
- All **1319 GSM8K** and **500 MATH-500** pinned reference answers parse.
- Downloaded only the official tokenizer, verified its prompt opens `<think>`,
  and checked its special-token behavior. Regression examples use the actual
  tokenizer to ensure intermediate boxed thoughts cannot supply a final answer.

## Independent review and fixes

A separate read-only reviewer found two Important scoring issues: removal of
`</think>` before grading, and LaTeX `\,` handling. Both were reproduced by
failing regression tests and fixed before the final green suite. No remaining
Critical/Important issues were identified in generation, quantization scope,
paired inputs, resume, packaging or notebook wiring. No deferred minor findings.

## Deliberate boundaries

- No full 1.4B weights or CUDA GPU were available for local inference. There is
  **no measured pretrained accuracy, Colab throughput or GPU memory result**.
- PTQ is NVFP4-format numerical QDQ, not a packed native low-bit kernel; native
  engine export equivalence and low-bit speedups were not tested or claimed.
- The source still computes all loops. Selected depth is a simulated-exit
  proxy; true variable-depth rollout/cache behavior is a future task.
- Research baselines require native BF16 CUDA. T4 is rejected, not relabeled.
- Math-Verify's Windows timeout worker uses an unpicklable nested function.
  Only explicitly invalid CPU tests disable grading timeouts. Formal MATH
  scoring uses Linux/Colab with 5-second parse/verify timeouts.
- Preserved final-answer boundaries intentionally reject unfinished thinking;
  inspect truncation and parse-failure rates before interpreting accuracy.

The historical `validation.md`, `design.md` and `implementation-plan.md`
describe the earlier Base-model starter, not the current experiment protocol.
