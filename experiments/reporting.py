"""Append-only problem results, strict resume identity and paired comparisons."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from .data import digest_json


def write_json(path, value):
    path = Path(path)
    # Same-directory replace prevents a partial manifest/summary after interruption.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_run(directory, protocol, resume=False):
    directory = Path(directory)
    path = directory / "manifest.json"
    identity = digest_json(protocol)
    if directory.exists() and any(directory.iterdir()):
        if not resume:
            raise FileExistsError(f"Output directory is not empty: {directory}; use --resume or a new directory")
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("protocol_sha256") != identity:
            raise ValueError("Resume protocol differs from the saved manifest (code, data, environment or settings changed)")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path, {"protocol_sha256": identity, "protocol": protocol})
    return identity


def read_records(directory):
    path = Path(directory) / "results.jsonl"
    if not path.exists():
        return []
    records = []
    # Fail clearly on an incomplete last line; never silently discard research data.
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Partial result at {path}:{number}; preserve the file and remove only that incomplete line before resuming") from error
    return records


def record_key(row):
    return row["arm"], row["dataset"], row["example_id"], row["threshold"]


def mean_se(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "se": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None}


def summarize(records):
    # Pre-early-exit records have no mode field and always executed all loops.
    modes = {row.get("execution_mode", "simulated") for row in records}
    if len(modes) > 1 or not modes <= {"simulated", "early_exit"}:
        raise ValueError("Cannot summarize mixed or unknown execution modes")
    mode = next(iter(modes), "simulated")
    groups = defaultdict(list)
    paired = defaultdict(dict)
    seen = set()
    for row in records:
        key = record_key(row)
        if key in seen:
            raise ValueError(f"Duplicate result: {key}")
        seen.add(key)
        groups[(row["dataset"], row["threshold"], row["arm"])].append(row)
        paired[(row["dataset"], row["threshold"])].setdefault(row["example_id"], {})[row["arm"]] = row
    summaries = []
    for (dataset, threshold, arm), rows in sorted(groups.items()):
        total_tokens = sum(r["generated_tokens"] for r in rows)
        summaries.append({
            "dataset": dataset, "threshold": threshold, "arm": arm, "examples": len(rows),
            "accuracy": mean_se(int(r["correct"]) for r in rows),
            "generated_tokens": mean_se(r["generated_tokens"] for r in rows),
            "mean_selected_depth_per_problem": mean_se(r["selected_core_steps"] / r["generated_tokens"] for r in rows),
            "selected_core_steps": mean_se(r["selected_core_steps"] for r in rows),
            "executed_core_steps": mean_se(r["executed_core_steps"] for r in rows),
            "mean_selected_depth_token_weighted": sum(r["selected_core_steps"] for r in rows) / total_tokens,
            "mean_executed_depth_token_weighted": sum(r["executed_core_steps"] for r in rows) / total_tokens,
            "selected_depth_histogram": dict(sorted(Counter(d for r in rows for d in r["selected_depths"]).items())),
            "executed_depth_histogram": dict(sorted(Counter(d for r in rows for d in r["executed_depths"]).items())),
            "truncation_rate": sum(r["truncated"] for r in rows) / len(rows),
            "parse_failure_rate": sum(not r["parse_success"] for r in rows) / len(rows),
            "generation_seconds": mean_se(r["generation_seconds"] for r in rows),
        })
    pairs = []
    for (dataset, threshold), by_id in sorted(paired.items()):
        matches = [(r["bf16"], r["ptq_nvfp4"]) for r in by_id.values() if "bf16" in r and "ptq_nvfp4" in r]
        solved = [(b, q) for b, q in matches if b["correct"] and q["correct"]]
        pair = {"dataset": dataset, "threshold": threshold, "paired_examples": len(matches),
                "jointly_solved": len(solved),
                "accuracy_delta_ptq_minus_bf16": mean_se(int(q["correct"]) - int(b["correct"]) for b, q in matches),
                "generated_tokens_delta_ptq_minus_bf16": mean_se(q["generated_tokens"] - b["generated_tokens"] for b, q in matches),
                "selected_compute_inflation_per_problem": mean_se(q["selected_core_steps"] / b["selected_core_steps"] - 1 for b, q in solved),
                "selected_compute_inflation_ratio_of_sums": (sum(q["selected_core_steps"] for b, q in solved)
                    / sum(b["selected_core_steps"] for b, q in solved) - 1) if solved else None,
                "executed_compute_inflation_ratio_of_sums": (sum(q["executed_core_steps"] for b, q in solved)
                    / sum(b["executed_core_steps"] for b, q in solved) - 1) if solved else None,
                "jointly_solved_with_truncation": sum(b["truncated"] or q["truncated"] for b, q in solved)}
        pairs.append(pair)
    semantics = ("physical early exit with last-exit KV fill; selected equals executed for generation decisions; prompt prefix excluded"
                 if mode == "early_exit" else "upstream simulated exit; selected is a proxy, executed runs all loops")
    return {"schema_version": 2, "execution_mode": mode, "depth_semantics": semantics,
            "groups": summaries, "paired": pairs}


def save_reports(directory, records):
    directory = Path(directory)
    report = summarize(records)
    write_json(directory / "summary.json", report)
    fields = ["dataset", "threshold", "arm", "execution_mode", "examples", "accuracy", "accuracy_se",
              "mean_generated_tokens", "mean_selected_depth", "mean_executed_depth",
              "mean_selected_core_steps", "mean_executed_core_steps", "truncation_rate", "parse_failure_rate"]
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in report["groups"]:
            row = {k: group[k] for k in ("dataset", "threshold", "arm", "examples", "truncation_rate", "parse_failure_rate")}
            row["execution_mode"] = report["execution_mode"]
            row.update(accuracy=group["accuracy"]["mean"], accuracy_se=group["accuracy"]["se"],
                       mean_generated_tokens=group["generated_tokens"]["mean"],
                       mean_selected_depth=group["mean_selected_depth_token_weighted"],
                       mean_executed_depth=group["mean_executed_depth_token_weighted"],
                       mean_selected_core_steps=group["selected_core_steps"]["mean"],
                       mean_executed_core_steps=group["executed_core_steps"]["mean"])
            writer.writerow(row)
    return report
