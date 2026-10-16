# Thinking Baselines Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run reproducible Thinking BF16 / MLP-only NVFP4 RTN comparisons in Colab.

**Architecture:** Keep the upstream model editable and instrument it externally.
Separate numerical quantization, generation traces, datasets/scoring and reports.
One CLI runs each arm sequentially, saves progress and produces paired summaries.

**Tech Stack:** Python, PyTorch >=2.6,<3, Transformers 4.57.1, HF Hub,
PyArrow, Math-Verify, Jupyter/Colab.

**Spec:** ../../baselines-design.md

## Global Constraints

Native BF16 CUDA; explicit CPU test mode only. Batch size 1. Core MLP only.
Preserve upstream source semantics, count selected and executed loops separately.
No real low-bit kernel speed claims. Fixed source/dataset revisions and run hashes.

## Review Focus

- First predicted token comes from prefill; EOS and truncation must count once.
- Equal-distance FP4 values round to even; tiny/zero blocks stay finite.
- Unparsed answers and truncated generations remain visible in reports.
- Resuming with changed code/questions/settings must fail before appending.
- Source ZIP and notebook must contain every new runtime module.

## Tasks

- [x] Add failing tests for RTN levels/scales and MLP-only mutation;
  implement `experiments/quantization.py`; verify CPU tests.
- [x] Add failing tests for cached generation/depth alignment against upstream;
  implement `experiments/generation.py`; verify 1/4 loops, EOS, token cap.
- [x] Add grading, pairing and resume tests; implement `experiments/scoring.py`,
  `experiments/data.py`, `experiments/reporting.py`, and `benchmark.py`.
- [x] Switch upstream/config metadata; update simple inference to Thinking chat;
  extend packaging and add `colab/Ouro_Thinking_Baselines.ipynb`.
- [x] Run complete unit/integration tests, tiny offline CPU benchmark, notebook
  upload and cell execution; document exact evidence and package fresh ZIP.

No Git repository exists in this project; do not create commits or alter parent
repositories. Existing source hashes are checked before replacing upstream files.

Execution record: 27 tests passed; source and real test-set reference checks
passed. Independent review's two scoring findings fixed with regressions. See
../../baselines-validation.md for evidence and limitations. No real GPU results.
