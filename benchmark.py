"""Compare Thinking BF16 and core-MLP NVFP4 RTN with paired generation metrics."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["gsm8k", "math500"], default=["gsm8k", "math500"])
    parser.add_argument("--limit", type=int, default=8, help="Examples per dataset; 0 = all test examples")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many examples after the fixed seeded shuffle, per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[1.0, .5])
    parser.add_argument("--execution-mode", choices=["early_exit", "simulated"], default="early_exit",
                        help="Physical stopping with last-exit KV fill, or upstream full-compute diagnostics")
    parser.add_argument("--loops", type=int, default=None, help="Default: editable local model config")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-file", type=Path, help="Custom/offline JSONL: dataset,id,question,answer")
    parser.add_argument("--allow-cpu-test", action="store_true", help="Force float32 CPU; explicitly INVALID as BF16 research data")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.offset < 0 or args.max_new_tokens < 1 or args.max_total_tokens <= args.max_new_tokens:
        parser.error("Invalid example limit or token budget")
    if args.loops is not None and args.loops < 1:
        parser.error("--loops must be positive")
    if any(not math.isfinite(t) or not 0 < t <= 1 for t in args.thresholds):
        parser.error("Exit thresholds must be in (0, 1]")
    if len(set(args.thresholds)) != len(args.thresholds) or len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate thresholds/datasets are not allowed")
    return args


def code_hashes():
    paths = [ROOT / name for name in ("benchmark.py", "inference.py", "upstream.json", "requirements.txt", "requirements-bench.txt")]
    for name in ("ouro_local", "experiments"):
        paths.extend(p for p in (ROOT / name).rglob("*") if p.suffix in (".py", ".json") and "__pycache__" not in p.parts)
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(args):
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed
    from inference import build_config
    from ouro_local.modeling_ouro import OuroForCausalLM
    from experiments.data import load_examples, encode_prompt, PROMPT_SUFFIX
    from experiments.generation import generate_traced
    from experiments.quantization import quantize_core_mlp_
    from experiments.scoring import grade_answer, grade_token_ids, SCORER_VERSION
    from experiments.reporting import prepare_run, read_records, record_key, save_reports, write_json

    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Install requirements-bench.txt: this project requires Transformers 4.57.1")
    if Path(inspect.getfile(OuroForCausalLM)).resolve() != ROOT / "ouro_local/modeling_ouro.py":
        raise RuntimeError("The benchmark must execute the uploaded local model source")
    if args.allow_cpu_test:
        device, dtype = "cpu", torch.float32
        torch.set_num_threads(2)
        print("INVALID_CPU_TEST_RUN: reference arm uses float32, not BF16. Do not report research accuracy.", flush=True)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Select a Colab GPU with native BF16 support (for example L4/A100)")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("This GPU lacks native BF16. T4/FP16 is not a BF16 baseline; use L4/A100 or another native BF16 GPU.")
        device, dtype = "cuda", torch.bfloat16
    if "math500" in args.datasets and os.name != "posix" and not args.allow_cpu_test:
        raise RuntimeError("MATH-500 research scoring requires Linux/Colab for Math-Verify's signal timeout")
    # Math-Verify's Windows spawn wrapper cannot pickle its nested worker.
    # Only explicitly invalid CPU checks may disable scoring timeouts.
    grading_timeout = 0 if args.allow_cpu_test else 5
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    upstream = json.loads((ROOT / "upstream.json").read_text(encoding="utf-8"))
    config = build_config(args.loops)
    tokenizer = AutoTokenizer.from_pretrained(upstream["model_id"], revision=upstream["revision"], trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Thinking evaluation requires a chat template")
    eos_ids = {tokenizer.eos_token_id}
    if "<|im_end|>" in tokenizer.get_vocab():
        eos_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    eos_ids.discard(None)
    if not eos_ids:
        raise ValueError("Tokenizer has no EOS token")
    examples, sources = load_examples(args.datasets, args.limit, args.seed, args.data_file, offset=args.offset)
    encoded = []
    limit = min(args.max_total_tokens, config.max_position_embeddings)
    for row in examples:
        prompt, ids = encode_prompt(tokenizer, row["question"])
        if ids.shape[1] + args.max_new_tokens > limit:
            raise ValueError(f"{row['dataset']}:{row['id']} needs {ids.shape[1]} prompt tokens + {args.max_new_tokens} output; exceeds {limit}. Increase context budget or reduce output. No examples were silently skipped.")
        # Validate gold answers and symbolic dependencies before spending GPU time.
        grade_answer(row["dataset"], "", row["answer"], timeout_seconds=grading_timeout)
        encoded.append((row, prompt, ids))
    versions = {}
    for package in ("torch", "transformers", "huggingface-hub", "tokenizers", "accelerate", "safetensors", "pyarrow", "math-verify", "latex2sympy2_extended", "sympy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    runtime = {"python": sys.version.split()[0], "packages": versions,
               "dtype": str(dtype).removeprefix("torch."), "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
               "attention": args.attention, "tf32": False}
    protocol = {
        "schema_version": 1, "model_id": upstream["model_id"], "revision": upstream["revision"],
        "config": config.to_dict(), "source_sha256": code_hashes(), "runtime": runtime,
        "research_valid": not args.allow_cpu_test, "datasets": sources, "examples": examples,
        "selection": {"method": "seeded_shuffle_slice_v1", "offset": args.offset, "limit": args.limit},
        "seed": args.seed, "thresholds": args.thresholds, "arms": ["bf16", "ptq_nvfp4"],
        "max_new_tokens": args.max_new_tokens, "max_total_tokens": args.max_total_tokens,
        "decoding": "greedy", "batch_size": 1, "chat_template": tokenizer.chat_template,
        "enable_thinking": True, "prompt_suffix": PROMPT_SUFFIX, "scorer": SCORER_VERSION,
        "grading_timeout_seconds": grading_timeout,
        "grading_requires_thinking_end": True,
        "execution_mode": args.execution_mode,
        "depth_mode": "physical_early_exit" if args.execution_mode == "early_exit" else "upstream_full_compute_simulated_exit",
        "adaptive_cache_policy": "last_executed_loop_kv_fill_v1" if args.execution_mode == "early_exit" else "full_depth_per_loop_kv",
        "prefill_policy": "full_depth_prefix_then_adaptive_last_prompt_token" if args.execution_mode == "early_exit" else "full_depth_entire_prompt",
        "threshold_one_prefill": "full_depth_entire_prompt",
        "trajectory_policy": "independent_generation_and_fresh_cache_per_precision_threshold_question",
        "threshold_one_policy": "explicit_final_loop_to_avoid_rounded_BF16_CDF_early_exit",
        "token_count": "all generated token IDs including reasoning and terminal EOS; excludes prompt",
        "quantization": "NVFP4 W4A16 MLP-only RTN QDQ; dequantized storage; no native low-bit GEMM",
    }
    output_dir = args.output_dir.resolve()
    identity = prepare_run(output_dir, protocol, args.resume)
    records = read_records(output_dir)
    expected_keys = {(arm, row["dataset"], str(row["id"]), threshold)
                     for arm in protocol["arms"] for threshold in args.thresholds for row in examples}
    completed = {record_key(row) for row in records}
    if len(completed) != len(records) or not completed <= expected_keys:
        raise ValueError("Duplicate or unexpected records in resume file")
    if any(row.get("protocol_sha256") != identity for row in records):
        raise ValueError("Result records belong to a different protocol")
    if completed == expected_keys:
        report = save_reports(output_dir, records)
        print(f"Already complete: {len(records)} records; {output_dir / 'summary.csv'}", flush=True)
        return report
    print(f"Weights: {upstream['model_id']} @ {upstream['revision']}\nRuntime: {runtime}\nResults: {output_dir}", flush=True)
    model = OuroForCausalLM.from_pretrained(
        upstream["model_id"], revision=upstream["revision"], config=config, dtype=dtype,
        device_map={"": device}, attn_implementation=args.attention).eval()
    for arm in protocol["arms"]:
        pending = [(row, prompt, ids, threshold) for threshold in args.thresholds for row, prompt, ids in encoded
                   if (arm, row["dataset"], str(row["id"]), threshold) not in completed]
        if not pending:
            continue
        if arm == "ptq_nvfp4":
            print("Applying MLP-only NVFP4 RTN QDQ (no packed inference)...", flush=True)
            quantization = quantize_core_mlp_(model)
            write_json(output_dir / "quantization.json", quantization)
        for row, prompt, ids, threshold in pending:
            set_seed(args.seed)
            trace = generate_traced(model, ids.to(device), threshold=threshold,
                                    max_new_tokens=args.max_new_tokens, eos_ids=eos_ids,
                                    execution_mode=args.execution_mode)
            grade = grade_token_ids(row["dataset"], tokenizer, trace["token_ids"], row["answer"],
                                    timeout_seconds=grading_timeout, require_thinking_end=True)
            record = {"protocol_sha256": identity, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                      "is_test_run": args.allow_cpu_test, "arm": arm, "dataset": row["dataset"],
                      "example_id": str(row["id"]), "threshold": threshold,
                      "question": row["question"], "gold_answer": row["answer"], "prompt": prompt,
                      **trace, **grade}
            with (output_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records.append(record)
            completed.add(record_key(record))
            save_reports(output_dir, records)
            print(f"[{len(completed)}/{len(expected_keys)}] {arm} {row['dataset']}:{row['id']} threshold={threshold} "
                  f"correct={grade['correct']} N={trace['generated_tokens']} "
                  f"gate_depth={trace['mean_selected_depth']:.3f} executed_depth={trace['mean_executed_depth']:.3f} "
                  f"stop={trace['stop_reason']}", flush=True)
    report = save_reports(output_dir, records)
    print(f"Complete. Summary: {output_dir / 'summary.csv'}", flush=True)
    return report


def main():
    try:
        run(parse_args())
    except (ValueError, RuntimeError, FileExistsError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
