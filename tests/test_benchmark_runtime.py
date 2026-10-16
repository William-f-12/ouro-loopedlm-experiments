"""Run the real CLI offline on a tiny checkpoint, including interruption resume."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_baselines import tiny_model


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_cli_runs_both_arms_and_resumes_without_duplicates(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "editable project"
            project.mkdir()
            for name in ("ouro_local", "experiments"):
                shutil.copytree(ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__"))
            for name in ("benchmark.py", "inference.py", "requirements.txt", "requirements-bench.txt"):
                shutil.copyfile(ROOT / name, project / name)
            weights = Path(tmp) / "tiny checkpoint"
            model = tiny_model()
            import torch
            with torch.no_grad():
                model.model.early_exit_gate.weight.zero_()
                model.model.early_exit_gate.bias.fill_(100)
            model.save_pretrained(weights)
            model.config.save_pretrained(project / "ouro_local")
            words = ["<eos>", "<unk>"] + [str(i) for i in range(30)]
            token = Tokenizer(WordLevel(dict(zip(words, range(32))), unk_token="<unk>"))
            token.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=token, eos_token="<eos>", unk_token="<unk>")
            tokenizer.chat_template = "{% for m in messages %}{{m['content']}}{% endfor %}"
            tokenizer.save_pretrained(weights)
            (project / "upstream.json").write_text(json.dumps({"model_id": str(weights), "revision": "offline-tiny"}))
            data = project / "tiny.jsonl"
            data.write_text('\n'.join(json.dumps({"id": str(i), "dataset": "gsm8k", "question": "1 + 1?", "answer": "#### 2"}) for i in range(2)), encoding="utf-8")
            output = project / "results"
            command = [sys.executable, str(project / "benchmark.py"), "--allow-cpu-test",
                       "--datasets", "gsm8k", "--data-file", str(data), "--limit", "2",
                       "--max-new-tokens", "3", "--max-total-tokens", "128", "--output-dir", str(output)]
            env = os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1"}
            def execute(extra):
                return subprocess.run(command + extra, cwd=project, env=env, capture_output=True,
                                      text=True, encoding="utf-8", timeout=180)
            run = execute([])
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            results = (output / "results.jsonl").read_bytes()
            rows = [json.loads(line) for line in results.splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["arm"] for r in rows}, {"bf16", "ptq_nvfp4"})
            self.assertTrue(all(r["is_test_run"] for r in rows))
            self.assertEqual({r["execution_mode"] for r in rows}, {"early_exit"})
            for row in rows:
                depth = 4 if row["threshold"] == 1 else 1
                self.assertEqual(row["executed_depths"], [depth] * row["generated_tokens"])
                self.assertEqual(row["selected_depths"], row["executed_depths"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["protocol"]["execution_mode"], "early_exit")
            self.assertEqual(manifest["protocol"]["adaptive_cache_policy"], "last_executed_loop_kv_fill_v1")
            self.assertFalse(manifest["protocol"]["research_valid"])
            self.assertEqual(manifest["protocol"]["runtime"]["dtype"], "float32")
            self.assertEqual(len(json.loads((output / "summary.json").read_text())["groups"]), 4)
            resume = execute(["--resume"])
            self.assertEqual(resume.returncode, 0, resume.stdout + resume.stderr)
            self.assertEqual((output / "results.jsonl").read_bytes(), results)
            changed = execute(["--resume", "--seed", "888"])
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("protocol", changed.stderr)
            changed_mode = execute(["--resume", "--execution-mode", "simulated"])
            self.assertNotEqual(changed_mode.returncode, 0)
            self.assertIn("protocol", changed_mode.stderr)
            simulated_dir = project / "simulated-results"
            simulated = execute(["--execution-mode", "simulated", "--output-dir", str(simulated_dir)])
            self.assertEqual(simulated.returncode, 0, simulated.stdout + simulated.stderr)
            simulated_rows = [json.loads(line) for line in (simulated_dir / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(simulated_rows), 8)
            for row in simulated_rows:
                self.assertEqual(row["execution_mode"], "simulated")
                self.assertEqual(row["executed_depths"], [4] * row["generated_tokens"])
                self.assertEqual(row["selected_depths"], [4 if row["threshold"] == 1 else 1] * row["generated_tokens"])
                if row["threshold"] == 1:
                    reference = next(r for r in rows if (r["arm"], r["example_id"], r["threshold"]) == (row["arm"], row["example_id"], 1.))
                    self.assertEqual(row["token_ids"], reference["token_ids"])
            second_output = project / "second-batch"
            second = execute(["--offset", "1", "--limit", "1", "--output-dir", str(second_output)])
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            second_manifest = json.loads((second_output / "manifest.json").read_text())
            self.assertEqual(second_manifest["protocol"]["selection"]["offset"], 1)
            self.assertEqual(second_manifest["protocol"]["examples"], manifest["protocol"]["examples"][1:])
            second_rows = (second_output / "results.jsonl").read_text().splitlines()
            self.assertEqual(len(second_rows), 4)


if __name__ == "__main__":
    unittest.main()
