import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_disabled_actual_cells_work_without_baseline_results_or_output(self):
        notebook = json.loads((ROOT/'colab/Ouro_RQ2_Probes.ipynb').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            scope = {'Path':Path, 'STORAGE_ROOT':Path(tmp), 'PROJECT_DIR':ROOT, 'sys':sys, 'subprocess':subprocess}
            # Actual parameter and command cells, not a duplicated command builder.
            exec(''.join(notebook['cells'][12]['source']), scope)
            exec(''.join(notebook['cells'][14]['source']), scope)
            self.assertFalse(scope['ENABLE_RQ2'])
            self.assertFalse(scope['RQ2_RESULT_DIR'].exists())
            self.assertFalse(scope['SOURCE_RESULTS'].exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_enabled_cell_builds_valid_probe_options_and_separate_paths(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('rq2_probe', ROOT/'scripts/rq2_probe.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        notebook = json.loads((ROOT/'colab/Ouro_RQ2_Probes.ipynb').read_text(encoding='utf-8'))
        scope = {'Path':Path, 'STORAGE_ROOT':Path('/content/drive/MyDrive/ouro-experiments'), 'PROJECT_DIR':ROOT, 'sys':sys}
        exec(''.join(notebook['cells'][12]['source']), scope)
        scope['ENABLE_RQ2'] = True
        tree = ast.parse(''.join(notebook['cells'][14]['source']))
        # Build command using real notebook logic; do not launch a full model in this test.
        tree.body = tree.body[:-1]
        exec(compile(tree, '<rq2-run-cell>', 'exec'), scope)
        args = module.parse_args(scope['command'][2:])
        self.assertTrue(args.enable_rq2)
        self.assertEqual(args.limit, 2)
        self.assertEqual(args.max_probe_tokens, 0)
        self.assertFalse(args.output_dir.is_relative_to(args.source_results))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<rq2-cell>', 'exec')
                self.assertEqual(cell['outputs'], [])


if __name__ == '__main__': unittest.main()
