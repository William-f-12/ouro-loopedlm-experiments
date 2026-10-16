"""Pinned public test sets, stable subset selection and explicit chat prompts."""
import hashlib
import json
from pathlib import Path
import random

DATASETS = {
    "gsm8k": {"repo_id": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866",
               "filename": "main/test-00000-of-00001.parquet", "split": "test", "config": "main"},
    "math500": {"repo_id": "HuggingFaceH4/MATH-500", "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                "filename": "test.jsonl", "split": "test"},
}
PROMPT_SUFFIX = "\n\nSolve this problem step by step. Put your final answer in \\boxed{} ."


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def load_examples(names, limit, seed, local_file=None, *, offset=0):
    """Return frozen examples and provenance; local JSONL is for offline/custom checks."""
    if limit < 0:
        raise ValueError("limit must be >= 0; 0 means the full test set")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    sources = {}
    if local_file:
        path = Path(local_file)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if row.get("dataset") not in DATASETS or not all(k in row for k in ("id", "question", "answer")):
                raise ValueError("Each local row needs dataset (gsm8k/math500), id, question, answer")
        sources["local"] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path.resolve())}
    else:
        from huggingface_hub import hf_hub_download
        rows = []
        for name in names:
            spec = DATASETS[name]
            path = Path(hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                       filename=spec["filename"], repo_type="dataset"))
            sources[name] = spec | {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if name == "gsm8k":
                import pyarrow.parquet as pq
                data = pq.read_table(path).to_pylist()
            else:
                data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for index, row in enumerate(data):
                rows.append({"dataset": name, "id": str(index),
                             "question": row["question" if name == "gsm8k" else "problem"],
                             "answer": row["answer"]})
    selected = []
    for name in names:
        group = [row for row in rows if row["dataset"] == name]
        if not group:
            raise ValueError(f"No examples found for {name}")
        if offset >= len(group):
            raise ValueError(f"offset {offset} is outside {name} ({len(group)} examples)")
        random.Random(f"{seed}:{name}").shuffle(group)
        selected.extend(group[offset:offset + limit] if limit else group[offset:])
    keys = [(row["dataset"], str(row["id"])) for row in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate example IDs")
    return selected, sources


def encode_prompt(tokenizer, question):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt",
                    return_token_type_ids=False).input_ids
    return prompt, ids
