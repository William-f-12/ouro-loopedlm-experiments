"""Unified source distribution and historical baseline upload compatibility."""
import contextlib
import importlib.util
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


class UnifiedProjectTests(unittest.TestCase):
    def bundle(self, destination):
        spec = importlib.util.spec_from_file_location('unified_packager', ROOT/'scripts/package_for_colab.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.create_bundle(ROOT, destination)

    def test_one_bundle_includes_baseline_and_rq2_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.bundle(Path(tmp))
            self.assertTrue(bundle.name.startswith('ouro-loopedlm-experiments-'))
            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
                for relative in ('benchmark.py', 'experiments/rq2.py', 'scripts/rq2_probe.py',
                                 'baseline_snapshot.json', 'RQ2_README.md', 'docs/project-layout.md'):
                    self.assertIn('ouro-loopedlm-experiments/' + relative, names)

    def test_all_notebooks_accept_new_and_historical_bundle_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            bundle = self.bundle(workspace/'bundles')
            for archive_root in ('ouro-loopedlm-experiments', 'ouro-1.4b-colab'):
                buffer = io.BytesIO()
                with zipfile.ZipFile(bundle) as original, zipfile.ZipFile(buffer, 'w') as target:
                    for name in original.namelist():
                        target.writestr(archive_root + '/' + name.split('/', 1)[1], original.read(name))
                for path in sorted((ROOT/'colab').glob('*.ipynb')):
                    with self.subTest(notebook=path.name, root=archive_root):
                        notebook = json.loads(path.read_text(encoding='utf-8'))
                        cell = next(c for c in notebook['cells'] if c['cell_type'] == 'code' and 'uploaded = files.upload()' in ''.join(c['source']))
                        fake = types.ModuleType('google.colab')
                        fake.files = types.SimpleNamespace(upload=lambda: {'source.zip': buffer.getvalue()})
                        namespace = {}
                        with patch.dict(sys.modules, {'google.colab': fake}), patch.dict(os.environ, {'OURO_COLAB_WORKSPACE': tmp}), contextlib.redirect_stdout(io.StringIO()):
                            exec(''.join(cell['source']), namespace)
                        self.assertEqual(namespace['PROJECT_DIR'].name, archive_root)
                        self.assertEqual((namespace['PROJECT_DIR']/'ouro_local/modeling_ouro.py').read_bytes(),
                                         (ROOT/'ouro_local/modeling_ouro.py').read_bytes())


if __name__ == '__main__':
    unittest.main()
