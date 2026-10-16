"""Validate disjoint completed batches and recompute summaries from raw records."""
import hashlib
import json
from pathlib import Path
import zipfile

from .data import DATASETS, digest_json
from .reporting import record_key, save_reports, summarize, write_json


def _read_batch(path):
    names = ("manifest.json", "results.jsonl", "quantization.json")
    if path.is_dir():
        return {name: (path / name).read_bytes() for name in names}
    # Downloaded notebook result ZIPs put these files at the archive root.
    # Read the entries directly: no extraction of uploaded paths or execution.
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError(f"Duplicate ZIP entries: {path}")
        return {name: archive.read(name) for name in names}


def merge_runs(inputs, output_dir, *, dataset="math500", expected_examples=500):
    """Require identical protocols except examples/selection; never average batch summaries."""
    if dataset not in DATASETS or expected_examples < 1 or not inputs:
        raise ValueError("Choose a dataset, positive expected example count, and at least one input")
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new directory")
    reference = None
    examples = {}
    records = []
    evidence = []
    for input_path in inputs:
        path = Path(input_path).resolve()
        if path == output:
            raise ValueError("Output must differ from all inputs")
        payload = _read_batch(path)
        manifest = json.loads(payload["manifest.json"])
        protocol = manifest["protocol"]
        identity = manifest["protocol_sha256"]
        if digest_json(protocol) != identity:
            raise ValueError(f"Manifest protocol hash mismatch: {path}")
        shared = {key: value for key, value in protocol.items() if key not in ("examples", "selection")}
        if reference is None:
            reference = shared
        elif shared != reference:
            changed = sorted(key for key in shared.keys() | reference.keys()
                             if shared.get(key) != reference.get(key))
            raise ValueError(f"Incompatible batch protocol: {path}; changed fields: {', '.join(changed)}")
        batch_examples = {}
        for example in protocol["examples"]:
            key = example["dataset"], str(example["id"])
            if key[0] != dataset:
                raise ValueError(f"Expected only {dataset}, found {key[0]}: {path}")
            if key in batch_examples or key in examples:
                raise ValueError(f"Overlapping/duplicate example {key}: {path}")
            batch_examples[key] = example
        if not batch_examples:
            raise ValueError(f"Empty batch: {path}")
        if protocol["arms"] != ["bf16", "ptq_nvfp4"]:
            raise ValueError(f"Expected the BF16/PTQ comparison protocol: {path}")
        thresholds = protocol["thresholds"]
        if not thresholds or len(set(thresholds)) != len(thresholds):
            raise ValueError(f"Invalid threshold protocol: {path}")
        expected = {(arm, name, example_id, threshold) for name, example_id in batch_examples
                    for arm in protocol["arms"] for threshold in thresholds}
        rows, found = [], set()
        for number, line in enumerate(payload["results.jsonl"].decode("utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Incomplete/invalid JSONL in {path} at line {number}; resume the batch first") from error
            if row.get("protocol_sha256") != identity:
                raise ValueError(f"Record protocol identity mismatch: {path}:{number}")
            key = record_key(row)
            if key in found:
                raise ValueError(f"Duplicate result {key}: {path}")
            if key not in expected:
                raise ValueError(f"Unexpected result {key}: {path}")
            example = batch_examples[row["dataset"], row["example_id"]]
            if row["question"] != example["question"] or row["gold_answer"] != example["answer"]:
                raise ValueError(f"Result question/answer differs from manifest: {path}:{number}")
            if row.get("is_test_run") != (not protocol["research_valid"]):
                raise ValueError(f"Record research validity differs from protocol: {path}:{number}")
            found.add(key)
            rows.append(row)
        if found != expected:
            raise ValueError(f"Incomplete batch {path}: missing {len(expected - found)} of {len(expected)} responses; resume it first")
        examples.update(batch_examples)
        records.extend(rows)
        evidence.append((path, payload, identity))
    if len(examples) != expected_examples:
        raise ValueError(f"Expected {expected_examples} unique {dataset} examples, found {len(examples)}")
    if dataset == "math500" and expected_examples == 500 and set(examples) != {("math500", str(i)) for i in range(500)}:
        raise ValueError("Full MATH-500 coverage requires exactly example IDs 0 through 499")
    # Validate the statistical fields before creating any output files.
    summarize(records)
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "provenance"
    provenance.mkdir()
    inputs_manifest = []
    for index, (path, payload, identity) in enumerate(evidence, 1):
        for name in ("manifest.json", "quantization.json"):
            (provenance / f"batch-{index:02d}-{name}").write_bytes(payload[name])
        inputs_manifest.append({"source": str(path), "protocol_sha256": identity,
            "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}})
    with (output / "results.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in records:
            # Keep original per-batch hashes for traceability; do not pretend this is a resumable run.
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "merge_manifest.json", {
        "schema_version": 1, "kind": "merged_completed_batches", "dataset": dataset,
        "unique_examples": len(examples), "responses": len(records), "shared_protocol": reference,
        "inputs": inputs_manifest,
    })
    return save_reports(output, records)
