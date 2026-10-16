"""Offline checks for non-overlapping batches and statistics from raw records."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.data import DATASETS, digest_json, load_examples
from experiments.reporting import read_records, summarize, write_json


class BatchSelectionTests(unittest.TestCase):
    def test_five_slices_cover_all_500_once_and_keep_the_original_first_100(self):
        import inspect
        self.assertIn("offset", inspect.signature(load_examples).parameters)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "math.jsonl"
            path.write_text("\n".join(json.dumps(dict(dataset="math500", id=str(i),
                question=f"Question {i}", answer=str(i))) for i in range(500)), encoding="utf-8")
            full, provenance = load_examples(["math500"], 0, 42, path)
            first, _ = load_examples(["math500"], 100, 42, path)
            batches = [load_examples(["math500"], 100, 42, path, offset=i * 100)[0] for i in range(5)]
            self.assertTrue(all(len(batch) == 100 for batch in batches))
            self.assertEqual(batches[0], first)
            combined = [row for batch in batches for row in batch]
            self.assertEqual(combined, full)
            self.assertEqual({row["id"] for row in combined}, {str(i) for i in range(500)})
            tail, tail_provenance = load_examples(["math500"], 0, 42, path, offset=400)
            self.assertEqual(tail, batches[-1])
            self.assertEqual(tail_provenance, provenance)
            for offset in (-1, 500, 501):
                with self.subTest(offset=offset), self.assertRaises(ValueError):
                    load_examples(["math500"], 100, 42, path, offset=offset)


class BatchMergeTests(unittest.TestCase):
    def merge(self, *args, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec("experiments.merging"), "Batch merger is not implemented")
        from experiments.merging import merge_runs
        return merge_runs(*args, **kwargs)

    def fixtures(self, root):
        directories, all_rows = [], []
        for i in range(5):
            path = root / f"batch-{i + 1}"
            path.mkdir()
            example = dict(dataset="math500", id=str(i), question=f"Question {i}", answer=str(i))
            protocol = dict(schema_version=1, model_id="offline-fixture", revision="fixture-v1",
                config={"total_ut_steps": 4}, seed=42, thresholds=[1.0, .5],
                arms=["bf16", "ptq_nvfp4"], max_new_tokens=2048, max_total_tokens=4096,
                research_valid=False, runtime={"device": "CPU", "dtype": "float32", "packages": {}},
                source_sha256={"experiments/reporting.py": hashlib.sha256((ROOT / "experiments/reporting.py").read_bytes()).hexdigest()},
                scorer="fixture-scorer", quantization="fixture-qdq", decoding="greedy",
                datasets={"math500": DATASETS["math500"] | {"sha256": "fixture"}},
                selection={"method": "seeded_shuffle_slice_v1", "offset": i, "limit": 1},
                examples=[example])
            identity = digest_json(protocol)
            write_json(path / "manifest.json", {"protocol": protocol, "protocol_sha256": identity})
            rows = []
            for arm in protocol["arms"]:
                for threshold in protocol["thresholds"]:
                    n = 2 * (i + 1) + (2 if arm == "ptq_nvfp4" else 0)
                    depth = 3 if arm == "ptq_nvfp4" else 2
                    rows.append(dict(protocol_sha256=identity, is_test_run=True,
                        dataset="math500", example_id=str(i), arm=arm, threshold=threshold,
                        question=example["question"], gold_answer=example["answer"],
                        correct=arm == "bf16" or i != 1, generated_tokens=n,
                        selected_core_steps=n * depth, executed_core_steps=n * 4,
                        selected_depths=[depth] * n, executed_depths=[4] * n,
                        truncated=False, parse_success=True, generation_seconds=n / 10))
            self.write_rows(path, rows)
            write_json(path / "quantization.json", {"format": "fixture"})
            # Deliberately stale summaries must never be used for merging.
            write_json(path / "summary.json", {"accuracy": 999})
            directories.append(path)
            all_rows.extend(rows)
        return directories, all_rows

    def write_rows(self, directory, rows):
        (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def change_protocol(self, directory, change):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        change(manifest["protocol"])
        manifest["protocol_sha256"] = digest_json(manifest["protocol"])
        write_json(directory / "manifest.json", manifest)
        rows = read_records(directory)
        for row in rows:
            row["protocol_sha256"] = manifest["protocol_sha256"]
        self.write_rows(directory, rows)

    def test_merging_recomputes_global_pairs_and_token_weighted_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, records = self.fixtures(root)
            output = root / "merged"
            report = self.merge(paths, output, dataset="math500", expected_examples=5)
            self.assertEqual(report, summarize(records))
            groups = {(g["arm"], g["threshold"]): g for g in report["groups"]}
            self.assertEqual(groups["bf16", .5]["generated_tokens"]["mean"], 6)
            self.assertEqual(groups["ptq_nvfp4", .5]["generated_tokens"]["mean"], 8)
            self.assertEqual(groups["ptq_nvfp4", .5]["accuracy"]["mean"], .8)
            self.assertEqual(groups["ptq_nvfp4", .5]["mean_selected_depth_token_weighted"], 3)
            for pair in report["paired"]:
                self.assertEqual(pair["paired_examples"], 5)
                self.assertEqual(pair["jointly_solved"], 4)
                self.assertAlmostEqual(pair["selected_compute_inflation_ratio_of_sums"], 102 / 52 - 1)
                self.assertAlmostEqual(pair["selected_compute_inflation_per_problem"]["mean"], 1.16875)
            self.assertEqual(len(read_records(output)), 20)
            merged = json.loads((output / "merge_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(merged["unique_examples"], 5)
            self.assertEqual(len(merged["inputs"]), 5)
            with self.assertRaises(FileExistsError):
                self.merge(paths, output, dataset="math500", expected_examples=5)

    def test_rejects_overlapping_batches_even_if_records_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, _ = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "[Dd]uplicate|[Oo]verlap"):
                self.merge(paths + [paths[0]], root / "merged", expected_examples=5)
            self.assertFalse((root / "merged").exists())

    def test_rejects_incomplete_batch_and_wrong_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, _ = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "500|[Cc]ount|[Ee]xpected"):
                self.merge(paths, root / "merged", expected_examples=500)
            self.write_rows(paths[-1], read_records(paths[-1])[:-1])
            with self.assertRaisesRegex(ValueError, "[Ii]ncomplete|[Mm]issing"):
                self.merge(paths, root / "merged", expected_examples=5)
            self.assertFalse((root / "merged").exists())

    def test_rejects_changed_protocol_including_runtime_and_model_code(self):
        mutations = [lambda p: p.update(seed=7), lambda p: p.update(max_new_tokens=4096),
            lambda p: p.update(scorer="different"), lambda p: p.update(quantization="different"),
            lambda p: p.update(execution_mode="early_exit", adaptive_cache_policy="last_executed_loop_kv_fill_v1"),
            lambda p: p["runtime"].update(device="different"),
            lambda p: p["source_sha256"].update({"model.py": "changed"})]
        for change in mutations:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths, _ = self.fixtures(root)
                self.change_protocol(paths[-1], change)
                with self.assertRaisesRegex(ValueError, "[Pp]rotocol|[Ss]ettings|[Cc]ompatib"):
                    self.merge(paths, root / "merged", expected_examples=5)
                self.assertFalse((root / "merged").exists())

    def test_rejects_tampered_manifest_or_record_identity(self):
        for corrupt_manifest in (True, False):
            with self.subTest(manifest=corrupt_manifest), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths, _ = self.fixtures(root)
                if corrupt_manifest:
                    manifest = json.loads((paths[0] / "manifest.json").read_text())
                    manifest["protocol_sha256"] = "wrong"
                    write_json(paths[0] / "manifest.json", manifest)
                else:
                    rows = read_records(paths[0])
                    rows[0]["protocol_sha256"] = "wrong"
                    self.write_rows(paths[0], rows)
                with self.assertRaisesRegex(ValueError, "[Hh]ash|[Ii]dentity|[Pp]rotocol"):
                    self.merge(paths, root / "merged", expected_examples=5)

    def test_cli_accepts_downloaded_result_zips_without_gpu_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, _ = self.fixtures(root)
            inputs = []
            for path in paths:
                zip_path = path.with_suffix(".zip")
                with zipfile.ZipFile(zip_path, "w") as archive:
                    for file in path.iterdir():
                        archive.write(file, file.name)
                inputs.append(str(zip_path))
            script = ROOT / "scripts/merge_results.py"
            self.assertTrue(script.is_file(), "Merge CLI is not implemented")
            output = root / "merged"
            result = subprocess.run([sys.executable, str(script), "--inputs", *inputs,
                "--dataset", "math500", "--expected-examples", "5", "--output-dir", str(output)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(read_records(output)), 20)
            self.assertTrue((output / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
