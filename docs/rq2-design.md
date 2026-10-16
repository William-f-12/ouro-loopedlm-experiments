# Optional RQ2 replay probes

User scope: independent project copy; original baseline code/configuration and
ongoing Colab run remain untouched. No pass@k, sampling, QAT or RL.

An independent notebook and CLI read a completed baseline result directory or
ZIP. `ENABLE_RQ2=False` / omission of `--enable-rq2` performs no model loading,
data reading or output writes. The existing benchmark and its notebook stay
byte-identical in this copy. A dedicated packager produces an RQ2-labelled ZIP.

For a fixed, seeded subset of problems, use each recorded BF16/PTQ trajectory
and threshold as an anchor. Both weight precisions teacher-force the SAME
prompt/token prefix AND recorded exit-depth schedule. They have separate
numerical KV caches. Diagnostic current-token forwards always complete four
loops. After probing, overwrite only the skipped loops' current-position KV
with the anchor's recorded last-exited KV before the next token. Simulated
anchors retain all full-depth KV. Reproduce the anchor's prefill policy.

Record post-core-RMSNorm hidden states (before the exit gate/LM head), per-loop
top-1 IDs, earliest suffix-constant top-1 loop, candidate gate exit, recorded
exit and signed exit-minus-stability gap. For the trajectory's own precision,
assert replayed gate depth and selected top-1 match the saved rollout. Fail
closed on mismatch. Later-loop predictions are counterfactual after a real exit;
top-1 stability does not imply correctness. The cross-precision candidate gate
is conditional on the anchor history, not an independently generated policy.

Pair states only within an anchor/token/loop. Report absolute L2 and relative
L2 (null for zero reference norm). Full histories are consumed, not independently
truncated for each precision. Optional max-token cap explicitly limits a prefix
and is recorded as diagnostic censoring. Default 0 analyzes all saved tokens.

One model on GPU: stream BF16 references to temporary CPU tensor files, apply
unchanged MLP-only RTN once, stream PTQ replays and pair comparisons. Final
JSONL holds scalar/token diagnostics, not hidden tensors. Summary groups never
mix anchor arm, threshold, execution mode or replay precision; SEs use problem
means rather than treating correlated tokens as independent observations.

Separate output directory, provenance and diagnostic wall time. Never rewrite
baseline results or call probe timing baseline throughput. An interrupted probe
uses a new output directory; automatic probe resume is out of scope.

Tests: hand-checked stability/error cases, both cache modes/thresholds,
variable-depth histories, replay parity on real tiny models, zero self-error,
nonzero PTQ error, invalid source rejection, disabled no-op, offline CLI end to
end, notebook execution wiring, entire inherited suite, original file hashes.
