# True early exit implementation

User decision: generate independent trajectories at each threshold. Threshold 1
always executes every loop; thresholds in (0, 1) stop at the first cumulative
conditional exit probability crossing, with the configured loop count as cap.

## Contract

- Add an opt-in, batch-one, single-token cached inference path to local Ouro.
  Default forward behavior remains compatible with the pinned upstream source.
- After a token exits, append that token's last executed loop K/V to every
  skipped loop's corresponding layer. Keep old positions unchanged; keys already
  contain RoPE. This last-exit fill is an approximation, recorded in provenance.
- Prefill all but the final prompt token at full depth, then generate the first
  output via an adaptive forward of the final prompt token. Fixed depth keeps
  the existing full-prompt prefill. Never count prefix work as output tokens.
- Each precision/threshold starts a fresh cache. Keep quantization, datasets,
  batching, token budgets, seed and scoring unchanged.
- Default benchmark/notebook to early_exit. Retain simulated mode for diagnostics.
  Record mode/cache/prefill policies, reject incompatible resumes/merges, and use
  a new run prefix. Unobserved exit probabilities are null, never invented.

## Steps and verification

- [x] Test and implement physical loop stopping and cache fill. Compare a sequence
  of changing depths to an independent full-compute oracle with manually filled
  KV, for both supported caches; count real decoder calls; test BF16 threshold 1.
- [x] Integrate traces, CLI, reporting and notebook. Verify independent caches,
  first-token behavior, EOS, real compute accounting, errors and mode provenance.
- [x] Update documentation, run existing and new tests plus smoke/notebook
  validation; obtain fresh review and package a clean upload ZIP.

Review focus: BF16 cumulative rounding, dynamic cache layer indexing, skipped
positions versus historical positions, first generation decision, mode mixing,
and preservation of the fixed-depth reference. No full checkpoint/GPU is locally
available; CPU tiny-model tests establish implementation behavior, not accuracy
or wallclock acceleration on Colab.

## Progress / decisions

- This directory is not a Git repository. Implement directly in the user's
  existing editable project; retain old bundles. No artificial branch/commit.
- Core true_early_exit consumes one token; generation handles prefill explicitly.
  Generic generate_traced retains simulated default for API compatibility;
  benchmark and notebook pass early_exit explicitly.

- Task 1 complete: four initial tests failed on missing mode and four-loop
  execution, then passed after physical stop/cache implementation. Extended
  threshold/loop-count/BF16 and error-cleanup checks pass.
- Task 2 complete: reporting mode and notebook/CLI integration tests failed
  before wiring, then passed. Full suite: 41/41, 51.314 s; smoke PASS. Both
  notebook schemas and code verified; ten unchanged baseline files checked
  against the previous source ZIP.
- Task 3 complete: documentation updated; fresh read-only review found no
  Critical/Important/actionable Minor findings, with an additional implicit
  cache transition probe passing. Packaging uses the existing deterministic
  source-only builder and per-file SHA256 verification. No pretrained/GPU
  accuracy or speed claim; cache-fill approximation is explicit in the guide.
