"""Validate CLI guardrails without loading or downloading a model."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InferenceArgumentsTests(unittest.TestCase):
    def load_entrypoint(self):
        path = ROOT / "inference.py"
        self.assertTrue(path.is_file(), "The inference entrypoint has not been implemented")
        spec = importlib.util.spec_from_file_location("inference_entrypoint", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_defaults_preserve_editable_config_and_limit_context(self):
        module = self.load_entrypoint()
        args = module.parse_args([])
        self.assertIsNone(args.loops)
        self.assertIsNone(args.early_exit_threshold)
        self.assertEqual(args.max_total_tokens, 4096)
        self.assertEqual(args.device, "cuda")
        self.assertTrue(args.chat)

    def test_invalid_experiment_parameters_fail_before_loading(self):
        module = self.load_entrypoint()
        for argv in [["--loops", "0"], ["--max-new-tokens", "-1"],
                     ["--temperature", "nan"], ["--top-p", "0"],
                     ["--early-exit-threshold", "1.5"],
                     ["--max-total-tokens", "16", "--max-new-tokens", "32"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
