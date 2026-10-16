"""Tests for the local-edit -> upload ZIP contract (standard library only)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class BundleTests(unittest.TestCase):
    def load_tools(self):
        path = ROOT / "scripts" / "package_for_colab.py"
        self.assertTrue(path.is_file(), "The source bundler has not been implemented")
        spec = importlib.util.spec_from_file_location("bundle_tools", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_modified_sources_are_bundled_without_weights_or_secrets(self):
        tools = self.load_tools()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project with spaces"
            for name, content in {
                "inference.py": "print('local edit')\n",
                "benchmark.py": "# baseline entrypoint\n",
                "requirements-bench.txt": "-r requirements.txt\n",
                "experiments/quantization.py": "# local quantization\n",
                "requirements.txt": "transformers==4.57.1\n",
                "ouro_local/__init__.py": "",
                "ouro_local/modeling_ouro.py": "LOCAL_EDIT = 42\n",
                "ouro_local/configuration_ouro.py": "# config\n",
                "ouro_local/config.json": "{}",
                "ouro_local/experiment/new_layer.py": "# new experiment\n",
                "ouro_local/__pycache__/cached.pyc": "cache",
                "ouro_local/.env": "SECRET=do-not-upload",
                "ouro_local/model.safetensors": "weight",
                ".env": "SECRET=do-not-upload",
                "outputs/result.json": "private result",
                ".venv/lib/secret.py": "private",
            }.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            bundle = tools.create_bundle(root)
            with zipfile.ZipFile(bundle) as archive:
                names = archive.namelist()
                source = "ouro-loopedlm-experiments/ouro_local/modeling_ouro.py"
                self.assertEqual(archive.read(source), (root / "ouro_local/modeling_ouro.py").read_bytes())
                self.assertIn("ouro-loopedlm-experiments/ouro_local/experiment/new_layer.py", names)
                self.assertIn("ouro-loopedlm-experiments/benchmark.py", names)
                self.assertIn("ouro-loopedlm-experiments/experiments/quantization.py", names)
                for excluded in [".env", "safetensors", "__pycache__", ".venv", "outputs/"]:
                    self.assertFalse(any(excluded in name for name in names), excluded)
                manifest = json.loads(archive.read("ouro-loopedlm-experiments/bundle_manifest.json"))
                self.assertIn("ouro_local/modeling_ouro.py", manifest["files"])
            self.assertEqual(bundle, tools.create_bundle(root))
            (root / "ouro_local/modeling_ouro.py").write_text("LOCAL_EDIT = 43\n")
            updated = tools.create_bundle(root)
            self.assertNotEqual(bundle.name, updated.name)


if __name__ == "__main__":
    unittest.main()
