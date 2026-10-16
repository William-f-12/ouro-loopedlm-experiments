"""Exercise the actual notebook upload cell with a simulated browser upload."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_for_colab import create_bundle


class NotebookTests(unittest.TestCase):
    def notebook(self):
        path = ROOT / "colab/Ouro_1_4B_Local.ipynb"
        self.assertTrue(path.is_file(), "Colab notebook has not been implemented")
        return json.loads(path.read_text(encoding="utf-8"))

    def upload(self, notebook, zip_bytes, workspace):
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == "upload-source")
        fake_colab = types.ModuleType("google.colab")
        fake_colab.files = types.SimpleNamespace(upload=lambda: {"edited.zip": zip_bytes})
        with patch.dict(sys.modules, {"google.colab": fake_colab}), \
             patch.dict(os.environ, {"OURO_COLAB_WORKSPACE": str(workspace)}), \
             contextlib.redirect_stdout(io.StringIO()):
            namespace = {}
            exec(compile("".join(cell["source"]), "notebook-upload", "exec"), namespace)
        return namespace["PROJECT_DIR"]

    def test_notebook_cells_compile_and_have_no_saved_outputs(self):
        for cell in self.notebook()["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_upload_preserves_sources_and_uses_fresh_directory(self):
        notebook = self.notebook()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = create_bundle(ROOT, root / "bundles")
            first = self.upload(notebook, bundle.read_bytes(), root)
            second = self.upload(notebook, bundle.read_bytes(), root)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "ouro_local/modeling_ouro.py").read_bytes(),
                             (ROOT / "ouro_local/modeling_ouro.py").read_bytes())
            self.assertTrue((first / "inference.py").is_file())

    def test_upload_rejects_modified_payload(self):
        notebook = self.notebook()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = create_bundle(ROOT, root / "bundles")
            corrupt = io.BytesIO()
            with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(corrupt, "w") as target:
                for member in source.infolist():
                    data = source.read(member)
                    if member.filename.endswith("/modeling_ouro.py"):
                        data += b"\n# changed after packaging\n"
                    target.writestr(member, data)
            with self.assertRaisesRegex(ValueError, "hash|Hash|SHA"):
                self.upload(notebook, corrupt.getvalue(), root)


class BaselineNotebookTests(NotebookTests):
    def notebook(self):
        path = ROOT / "colab/Ouro_Thinking_Baselines.ipynb"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_parameter_cell_and_run_cell_build_a_valid_benchmark_command(self):
        sys.path.insert(0, str(ROOT))
        from benchmark import parse_args
        calls = []
        namespace = {"Path": Path, "STORAGE_ROOT": ROOT / "outputs", "PROJECT_DIR": ROOT,
                     "sys": sys, "subprocess": types.SimpleNamespace(run=lambda argv, **kw: calls.append((argv, kw)))}
        cells = {cell["id"]: cell for cell in self.notebook()["cells"]}
        with contextlib.redirect_stdout(io.StringIO()):
            parameters = ast.parse("".join(cells["baseline-parameters"]["source"]))
            # Simulate selecting the fifth batch in the real parameter cell.
            for node in parameters.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BATCH_ID" for t in node.targets):
                    node.value = ast.Constant(value=5)
            exec(compile(ast.fix_missing_locations(parameters), "parameters", "exec"), namespace)
            namespace.update(LOOPS=3, RESUME=True)
            exec("".join(cells["run-baselines"]["source"]), namespace)
        command, kwargs = calls[0]
        self.assertEqual(Path(command[1]), ROOT / "benchmark.py")
        args = parse_args(command[2:])
        self.assertEqual(args.thresholds, [1., .5])
        self.assertEqual(args.datasets, ["math500"])
        self.assertEqual(args.offset, 400)
        self.assertEqual(args.limit, 100)
        self.assertEqual(args.output_dir.name, "baseline-math500-100q-early-exit-batch-05")
        self.assertEqual(args.execution_mode, "early_exit")
        self.assertEqual(namespace["EXECUTION_MODE"], "early_exit")
        self.assertIs(namespace["ENABLE_EARLY_EXIT"], True)
        self.assertIn("--execution-mode", command)
        self.assertEqual(args.max_new_tokens, 2048)
        self.assertEqual(args.loops, 3)
        self.assertTrue(args.resume)
        self.assertFalse(args.allow_cpu_test)
        self.assertEqual(kwargs["cwd"], ROOT)
        self.assertTrue(kwargs["check"])

    def test_switching_mode_automatically_separates_result_and_merge_paths(self):
        cells = {cell["id"]: cell for cell in self.notebook()["cells"]}
        paths = []
        for enabled, mode, prefix in (
            (True, "early_exit", "baseline-math500-100q-early-exit"),
            (False, "simulated", "baseline-math500-100q-simulated"),
        ):
            calls = []
            namespace = {"Path": Path, "STORAGE_ROOT": ROOT / "outputs", "PROJECT_DIR": ROOT,
                         "sys": sys, "subprocess": types.SimpleNamespace(run=lambda argv, **kw: calls.append(argv))}
            parameters = ast.parse("".join(cells["baseline-parameters"]["source"]))
            for node in parameters.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ENABLE_EARLY_EXIT" for t in node.targets):
                    node.value = ast.Constant(value=enabled)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.fix_missing_locations(parameters), "parameters", "exec"), namespace)
                exec("".join(cells["run-baselines"]["source"]), namespace)
            from benchmark import parse_args
            self.assertEqual(parse_args(calls[0][2:]).execution_mode, mode)
            paths.append(namespace["RESULT_DIR"])
            self.assertEqual(namespace["RUN_PREFIX"], prefix)
            # Execute the actual merge cell's path/command assignments without
            # reading result files or importing notebook display dependencies.
            merge = ast.parse("".join(cells["merge-batches"]["source"]))
            merge.body = [node for node in merge.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id in
                                  ("MERGE_INPUTS", "MERGED_DIR", "merge_command") for t in node.targets)]
            exec(compile(merge, "merge-paths", "exec"), namespace)
            self.assertEqual([p.name for p in namespace["MERGE_INPUTS"]],
                             [prefix + suffix for suffix in ("-batch-01", "-batch-02", "-batch-03", "-batch-04", "-batch-05")])
            self.assertEqual(namespace["MERGED_DIR"].name, prefix + "-merged-500")
        self.assertNotEqual(paths[0], paths[1])


if __name__ == "__main__":
    unittest.main()
