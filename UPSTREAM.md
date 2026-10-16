# Upstream provenance

Model: [ByteDance/Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking)

Pinned revision: `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` (checked 2026-10-11).
This replaces the previous plain Ouro-1.4B starter; historical results must not be pooled.

The initial `ouro_local/modeling_ouro.py`, `configuration_ouro.py` and
`config.json` are byte-for-byte copies of this revision. Their original URLs
and SHA256 digests are recorded in `upstream.json`. This file records the
original baseline; **do not update its source hashes when making experiments**.
The model card declares Apache-2.0. `LICENSE` contains the license, and the
configuration source retains its original Qwen / Hugging Face copyright notice.
The surrounding project helpers are also provided under Apache-2.0.

No model weights are included. Inference downloads the official weights and
tokenizer from the fixed revision, but imports the model class and configuration
from this project's `ouro_local` package. No Hub model Python code is executed
through `trust_remote_code`.

The wrapper supplies an explicit `transformers.DynamicCache()` for each new
request. The current Thinking source also contains upstream cache-mask and
default-RoPE fixes beyond the older plain-model source. Transformers 4.57.1 is
retained as the tested runtime.

- [Transformers version issue](https://huggingface.co/ByteDance/Ouro-1.4B/discussions/14)
- [Default cache mask issue](https://huggingface.co/ByteDance/Ouro-1.4B/discussions/10)

The upstream forward pass executes all `total_ut_steps` before selecting an
early-exit representation. Changing `early_exit_threshold` alone does not skip
these computations. Weight-sharing does not imply KV-cache sharing: a separate
cache slot is used for every loop/layer pair.

Local modification (2026-10-12): `modeling_ouro.py` now has an opt-in physical
loop stop (`true_early_exit=True`) for batch-one single-token cached inference.
The original source hashes in upstream.json are intentionally unchanged. All
other original model/config files remain unchanged. Default model forward and
the simple inference demo preserve upstream full computation.

The benchmark now defaults to independent trajectories with physical exit for
thresholds below 1. A token's last executed loop K/V is appended to matching
layers in skipped loops (already RoPE-rotated). This last-exit fill approximation
changes future deep-loop histories relative to upstream simulation. Prefill of
the prompt prefix is full-depth; the last prompt token uses adaptive inference.
At threshold 1 the benchmark retains its original full-prompt, final-loop path,
preventing BF16 CDF rounding from accidentally exiting earlier. The external
hook records actually observed gates; unexecuted probabilities are null.
`--execution-mode simulated` retains full-compute diagnostics. Manifests record
mode, fill and prefill policies; results from different protocols cannot mix.

The NVFP4 helper implements the published E2M1 / E4M3FN format with blocks of 16,
FP32 global scaling, and nearest/ties-to-even rounding. It materializes dequantized
weights and does not include native NVIDIA kernels or promise engine-export
compatibility. Actual quantized modules are listed in quantization.json.
