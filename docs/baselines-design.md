# Thinking BF16 / PTQ baseline design

Original design (2026-10-11), with early-exit update below. User-authorized scope: switch the existing editable project to
Ouro-1.4B-Thinking and compare accuracy, output tokens and loop depth in Colab.

- Pin Thinking weights, tokenizer, config and source to revision
  `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. Keep direct local imports.
- Add an evaluation CLI and a Colab baseline notebook. Evaluate GSM8K test and
  MATH-500 using pinned dataset revisions, deterministic seeded subsets,
  identical chat prompts and greedy decoding (one response per problem).
- Preserve upstream full-depth cache semantics. Default thresholds: 1.0 (fixed
  final-loop output) and 0.5 (gate-selected output). Record selected depth and
  actual executed depth separately; this is a simulated-exit baseline, not the
  proposal's future true early-exit rollout. Count the next-token decision at
  the last prompt position once, not once per prompt token.
- PTQ: RTN NVFP4 numerical reference, E2M1 weights, block size 16 along input
  features, E4M3 block scales, FP32 per-matrix scale. Only core MLP gate/up/down
  projections. Materialize dequantized BF16 weights once; activations and other
  weights remain BF16. No packed inference, QAT, GRPO or speedup claim.
- Require native BF16 CUDA for research baselines; T4 must not silently become
  an FP16 baseline. CPU float32 is available only as an explicitly marked test.
- Save per-token depth traces, per-problem answers/grades/censoring, environment,
  source hashes, selection and quantization manifests, paired differences and
  standard errors. Persist each completed problem and reject incompatible resumes.
- Default 8 examples per dataset, 2048 new tokens, batch size 1. Small defaults
  test the pipeline, not statistical significance. Exact numeric GSM8K grading;
  symbolic MATH grading through a pinned Math-Verify dependency. Preserve raw text.

Validation: hand-check FP4 rounding/scales; verify only MLP weights change;
compare traced generation with upstream `generate` on a tiny model; test EOS and
token-limit accounting, grading, pairing/resume; execute notebooks' upload and
CLI cells using a tiny offline checkpoint. Full pretrained GPU results require
the user's Colab runtime and must not be claimed as locally measured.

## Superseding exit design — 2026-10-12

The user explicitly chose independent threshold trajectories and physical stopping
when a threshold below 1 is met. The benchmark now defaults to `early_exit`,
retaining `simulated` for diagnostics. Fixed threshold 1 is unchanged. Adaptive
single-token inference uses cumulative conditional hazard probabilities, halts
the core loop, and fills skipped loop/layer cache entries with the token's last
executed loop K/V. Prompt prefix prefill stays full-depth; first output prediction
is adaptive. Selected and executed depth now agree, excluding prompt-prefix work.

See [implementation contract](superpowers/plans/2026-10-12-true-early-exit.md)
and [current experiment guide](baselines.md). The five-batch MATH-500 workflow,
BF16/PTQ scope, seed, budgets and grading are unchanged. Mode and cache policy
enter experiment identity; new and old trajectories must not be pooled.
