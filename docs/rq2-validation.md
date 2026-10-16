# RQ2 verification — 2026-10-12

## Executed evidence

- Full inherited + new suite: `python -B -m unittest discover -s tests -v`
  passed **57 tests**, no skips, in 298.225 seconds. That discovery started
  before the final scratch-path regression was added. The subsequently updated
  **six IO tests** also passed, including that new regression (58 distinct tests
  covered across these runs). CLI integration in the full suite exercised the
  corrected final guard implementation.
- Independent reviewer also ran the RQ2 tests and the six updated IO tests;
  no remaining substantive issue. The identified scratch isolation gap was
  reproduced, fixed before model allocation, and verified.
- Real tiny-checkpoint CLI generated a baseline, probed both precisions and both
  exit modes, checked successful provenance and per-loop metrics, verified source
  result bytes unchanged, and rejected overwrites. No network needed.
- FP32 and BF16 tiny models: original tokens/depths match replay, including
  one-token prompts, adaptive prefix prefill, simulated and fixed-depth paths.
- Variable recorded history 1/4/2/3 agrees exactly with independent physical
  early-exit states at all actually executed loops. No extra diagnostic KV leaks
  into the next token. Probing does not change later generation.
- Self-comparison yields zero L2; quantized weights yield nonzero errors.
  Hand-calculated stable suffix, L2/relative-L2, zero denominators and
  equal-problem-weight statistics pass.
- Disabled CLI imports no torch, reads no missing source and creates no output.
  Actual disabled notebook cells run successfully; enabled arguments parse with
  isolated paths. New notebook schema validates: 21 cells, no saved output.
- Existing `scripts/smoke_test.py` passed at loop caps 3 and 4, cached/full logits
  maximum differences 1.49e-7 and 1.04e-7.
- Original project's 47 recorded non-cache/non-verification file hashes remain
  unchanged, no new files there. Every inherited file in the copy is byte-identical.

## Environment and boundaries

Windows, Python 3.13.5, torch 2.6.0 CPU, Transformers 4.57.1. Existing isolated
dependencies were reused read-only; no global environment changes. Test temp
files were isolated in this project's `.verification/tmp` because the sandbox
system temporary directory denied writes. Normal Windows Math-Verify warnings
about disabled test-only timeouts were emitted by inherited tests.

No complete pretrained 1.4B GPU inference, actual Colab browser session, timing,
accuracy or RQ2 scientific outcome has been measured here. First Colab probe
should use the default two-problem subset and the matching baseline hardware /
versions. Own-precision token/depth replay mismatch fails explicitly.

Reference hidden tensors are temporary, one anchor at a time in CPU memory;
all BF16 anchors occupy local scratch until PTQ comparison. Full-token probes
need extra GPU time and local disk. Default limit=2 is a workflow check, not a
statistical result. No pass@k, QAT/RL or native packed NVFP4 kernel.
