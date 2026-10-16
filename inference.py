"""Run official Ouro weights through the editable local model implementation."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "ouro_local"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", default=None)
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True,
                        help="Thinking chat template (default); --no-chat for raw continuation")
    parser.add_argument("--loops", type=int, help="Override local config.total_ut_steps")
    parser.add_argument("--early-exit-threshold", type=float, help="Select an exit representation; does not skip loops")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int, default=4096, help="Input + requested output budget")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--no-cache", action="store_true", help="Diagnostic mode; slower for long generation")
    parser.add_argument("--output", type=Path, help="Write run metadata and generated text as JSON")
    args = parser.parse_args(argv)
    if args.loops is not None and args.loops < 1:
        parser.error("--loops must be positive")
    if args.max_new_tokens < 1 or args.max_total_tokens <= args.max_new_tokens:
        parser.error("Token budgets must be positive and leave room for input tokens")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and non-negative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.early_exit_threshold is not None and not 0 < args.early_exit_threshold <= 1:
        parser.error("--early-exit-threshold must be in (0, 1]")
    return args


def build_config(loops=None, early_exit_threshold=None):
    from ouro_local.configuration_ouro import OuroConfig
    config = OuroConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if loops is not None:
        config.total_ut_steps = loops
    if early_exit_threshold is not None:
        config.early_exit_threshold = early_exit_threshold
    if not isinstance(config.total_ut_steps, int) or config.total_ut_steps < 1:
        raise ValueError("ouro_local/config.json total_ut_steps must be a positive integer")
    return config


def source_provenance():
    paths = [ROOT / "inference.py", MODEL_DIR / "config.json", *sorted(MODEL_DIR.rglob("*.py"))]
    files = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths if "__pycache__" not in path.parts}
    return {"project_dir": str(ROOT), "sha256": files}


def run(args):
    import torch
    import transformers
    from transformers import AutoTokenizer, DynamicCache, set_seed
    # Direct local imports are intentional: NEVER replace these with remote AutoModel.
    from ouro_local.modeling_ouro import OuroForCausalLM

    if transformers.__version__ != "4.57.1":
        raise RuntimeError("This starter targets transformers==4.57.1. Install requirements.txt first.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. In Colab select Runtime > Change runtime type > GPU, then reconnect.")
    actual_source = Path(inspect.getfile(OuroForCausalLM)).resolve()
    if actual_source != (MODEL_DIR / "modeling_ouro.py").resolve():
        raise RuntimeError(f"Unexpected model source: {actual_source}")
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "float32" if args.device == "cpu" else (
            "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float16")
    if args.device == "cpu" and dtype_name == "float16":
        raise ValueError("Use float32 for CPU inference")
    if args.device == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This GPU has no native BF16 support; select auto or float16")
    dtype = getattr(torch, dtype_name)
    metadata = json.loads((ROOT / "upstream.json").read_text(encoding="utf-8"))
    config = build_config(args.loops, args.early_exit_threshold)
    prompt = (args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else
              args.prompt if args.prompt is not None else "Solve 2x + 3 = 11. Put the answer in \\boxed{}.")
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    provenance = source_provenance()
    print(f"Project: {ROOT}\nModel source: {actual_source}", flush=True)
    print(f"Source SHA256: {provenance['sha256']['ouro_local/modeling_ouro.py']}", flush=True)
    print(f"Weights: {metadata['model_id']} @ {metadata['revision']}", flush=True)
    print(f"Device: {args.device}; dtype: {dtype_name}; loops: {config.total_ut_steps}", flush=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_id"], revision=metadata["revision"], trust_remote_code=False)
    if args.chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=not args.chat,
                       return_token_type_ids=False)
    input_tokens = inputs.input_ids.shape[1]
    limit = min(args.max_total_tokens, config.max_position_embeddings)
    if input_tokens + args.max_new_tokens > limit:
        raise ValueError(f"Input ({input_tokens}) + requested output ({args.max_new_tokens}) exceeds {limit}. "
                         "Shorten the prompt/output, or explicitly increase --max-total-tokens with enough VRAM.")
    started = time.perf_counter()
    model = OuroForCausalLM.from_pretrained(
        metadata["model_id"], revision=metadata["revision"], config=config,
        dtype=dtype, device_map={"": args.device}, attn_implementation=args.attention,
    ).eval()
    load_seconds = time.perf_counter() - started
    inputs = inputs.to(args.device)
    eos_ids = [tokenizer.eos_token_id]
    if args.chat:
        chat_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if chat_end is not None and chat_end not in eos_ids:
            eos_ids.append(chat_end)
    generate_options = dict(
        max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=eos_ids,
        use_cache=not args.no_cache, logits_to_keep=1,
    )
    if not args.no_cache:
        generate_options["past_key_values"] = DynamicCache()
    if args.temperature > 0:
        generate_options.update(temperature=args.temperature, top_p=args.top_p)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **generate_options)
    if args.device == "cuda":
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    generated = output[0, input_tokens:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": metadata["model_id"], "revision": metadata["revision"],
        "source": provenance, "parameters": vars(args) | {"prompt_file": str(args.prompt_file) if args.prompt_file else None,
                                                         "output": str(args.output) if args.output else None},
        "effective_config": {"total_ut_steps": config.total_ut_steps,
                             "early_exit_threshold": config.early_exit_threshold},
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "transformers": transformers.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name() if args.device == "cuda" else "CPU", "dtype": dtype_name},
        "input_tokens": input_tokens, "generated_tokens": len(generated), "generated_text": text,
        "load_seconds": load_seconds, "generation_seconds": generation_seconds,
        "tokens_per_second": len(generated) / generation_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None,
    }
    print("\nGenerated text:\n" + text)
    print(f"\nGenerated {len(generated)} tokens in {generation_seconds:.2f}s")
    if result["peak_allocated_gib"] is not None:
        print(f"Peak allocated GPU memory: {result['peak_allocated_gib']:.2f} GiB")
    destination = args.output or ROOT / "outputs" / f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {destination.resolve()}")
    return result


def main():
    args = parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError) as error:
        print(f"Inference failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
