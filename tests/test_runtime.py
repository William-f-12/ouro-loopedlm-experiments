"""Full CLI integration with a tiny local checkpoint, never pretrained weights."""
import hashlib
import importlib.util
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
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None
                  for name in ("torch", "transformers", "tokenizers", "accelerate"))


@unittest.skipUnless(HAS_RUNTIME, "Install requirements.txt and PyTorch to run the offline CLI integration")
class RuntimeTests(unittest.TestCase):
    def test_cli_executes_edited_source_and_saves_run_metadata(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from ouro_local.configuration_ouro import OuroConfig
        from ouro_local.modeling_ouro import OuroForCausalLM

        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "edited project with spaces"
            project.mkdir()
            shutil.copytree(ROOT / "ouro_local", project / "ouro_local", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copyfile(ROOT / "inference.py", project / "inference.py")
            edited_source = project / "ouro_local/modeling_ouro.py"
            with edited_source.open("a", encoding="utf-8") as source:
                source.write('\nprint("LOCAL_EDIT_EXECUTED", flush=True)\n')
            config = OuroConfig(
                vocab_size=32, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                head_dim=16, max_position_embeddings=128, total_ut_steps=4,
                layer_types=["full_attention"] * 2, bos_token_id=0, eos_token_id=0, pad_token_id=0,
            )
            weights = root / "tiny_weights"
            model = OuroForCausalLM(config)
            model.save_pretrained(weights)
            config.save_pretrained(project / "ouro_local")
            words = ["<eos>", "<unk>", "The", "capital", "of", "France", "is", "Paris"]
            words += [f"word{i}" for i in range(32 - len(words))]
            tokenizer = Tokenizer(WordLevel(dict(zip(words, range(32))), unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<eos>", eos_token="<eos>", unk_token="<unk>")
            fast.save_pretrained(weights)
            (project / "upstream.json").write_text(json.dumps({"model_id": str(weights), "revision": "local-test"}))
            output = project / "result.json"
            env = os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1"}
            result = subprocess.run(
                [sys.executable, str(project / "inference.py"), "--device", "cpu", "--no-chat",
                 "--loops", "3", "--max-new-tokens", "4", "--output", str(output)],
                cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("LOCAL_EDIT_EXECUTED", result.stdout)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["effective_config"]["total_ut_steps"], 3)
            self.assertEqual(report["source"]["project_dir"], str(project))
            self.assertEqual(report["source"]["sha256"]["ouro_local/modeling_ouro.py"],
                             hashlib.sha256(edited_source.read_bytes()).hexdigest())
            self.assertGreater(report["generated_tokens"], 0)
            self.assertIsInstance(report["generated_text"], str)
            self.assertEqual(report["runtime"]["dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
