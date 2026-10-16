import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.data import digest_json


def fixture(directory):
    protocol = {'model_id': 'fixture', 'revision': 'fixture', 'config': {'total_ut_steps': 4},
                'arms': ['bf16', 'ptq_nvfp4'], 'thresholds': [1., .5],
                'execution_mode': 'early_exit', 'decoding': 'greedy', 'batch_size': 1,
                'research_valid': False, 'examples': [{'dataset': 'math500', 'id': str(i)} for i in range(3)]}
    identity = digest_json(protocol)
    records = []
    for i in range(3):
        for arm in protocol['arms']:
            for threshold in protocol['thresholds']:
                depth = 4 if threshold == 1 else 1
                records.append({'protocol_sha256': identity, 'dataset': 'math500', 'example_id': str(i),
                    'arm': arm, 'threshold': threshold, 'execution_mode': 'early_exit',
                    'prompt': 'p', 'token_ids': [7, 8], 'generated_tokens': 2,
                    'input_tokens': 1, 'selected_depths': [depth] * 2, 'executed_depths': [depth] * 2,
                    'correct': True})
    payload = {'manifest.json': {'protocol': protocol, 'protocol_sha256': identity},
               'quantization.json': {'format': 'NVFP4_E2M1_E4M3FN_RTN'}}
    for name, obj in payload.items():
        (directory / name).write_text(json.dumps(obj), encoding='utf-8')
    (directory / 'results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records), encoding='utf-8')
    return records


class AvailabilityTests(unittest.TestCase):
    def test_io_module_available(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.rq2_io'))

    def test_disabled_cli_does_not_read_input_create_output_or_load_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # Make loading the heavy runtime fail: a disabled command must not import it.
            (directory / 'torch.py').write_text('raise RuntimeError("must not import torch")')
            out = directory / 'should-not-exist'
            env = dict(os.environ, PYTHONPATH=str(directory))
            result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/rq2_probe.py'),
                '--source-results', str(directory/'missing'), '--output-dir', str(out)],
                capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('disabled', result.stdout.lower())
            self.assertFalse(out.exists())

    def test_scratch_path_cannot_write_inside_read_only_baseline(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('rq2_probe', ROOT/'scripts/rq2_probe.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'baseline'
            source.mkdir()
            fixture(source)
            before = {p.name:p.read_bytes() for p in source.iterdir()}
            # Validation must fail before missing source files/runtime are inspected.
            for scratch in (source, source/'scratch'):
                args = module.parse_args(['--enable-rq2', '--source-results', str(source),
                    '--output-dir', str(Path(tmp)/'probe'), '--scratch-dir', str(scratch)])
                with self.assertRaisesRegex(ValueError, 'scratch'):
                    module.run(args)
            self.assertEqual(before, {p.name:p.read_bytes() for p in source.iterdir()})


if importlib.util.find_spec('experiments.rq2_io') is not None:
    from experiments.rq2_io import load_anchors, summarize_probes

    class InputTests(unittest.TestCase):
        def test_summary_weights_problems_equally_and_never_pools_anchor_arms(self):
            def row(ident, depth, anchor='bf16'):
                info = {'candidate_exit_depth': depth, 'stable_from_loop': 1,
                        'exit_minus_stability': depth-1, 'extra_loops_after_stability': depth-1}
                return {'dataset':'math500', 'execution_mode':'early_exit', 'threshold':.5,
                        'anchor_arm':anchor, 'example_id':ident, 'bf16':info, 'ptq_nvfp4':info,
                        'hidden_error':{'l2_by_loop':[2.], 'relative_l2_by_loop':[None]}}
            data = [row('a', 1)] + [row('b', 3)] * 3 + [row('a', 4, 'ptq_nvfp4')]
            report = summarize_probes(data)
            self.assertEqual(len(report['groups']), 2)
            first = report['groups'][0]
            self.assertEqual(first['examples'], 2)
            self.assertEqual(first['token_decisions'], 4)
            self.assertEqual(first['metrics']['bf16_candidate_depth'], {'n':2, 'mean':2., 'se':1.})

        def test_directory_and_zip_select_same_complete_problem_pairs(self):
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                fixture(directory)
                _, a, _ = load_anchors(directory, limit=2, seed=42)
                archive = directory/'results.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    for name in ['manifest.json', 'results.jsonl', 'quantization.json']:
                        z.write(directory/name, name)
                _, b, _ = load_anchors(archive, limit=2, seed=42)
                self.assertEqual(a, b)
                self.assertEqual(len(a), 8)
                self.assertEqual(len({x['example_id'] for x in a}), 2)
                self.assertEqual({x['arm'] for x in a}, {'bf16', 'ptq_nvfp4'})

        def test_rejects_partial_duplicate_bad_depth_and_foreign_protocol(self):
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                for change in ['partial', 'duplicate', 'depth', 'foreign']:
                    rows = fixture(directory)
                    if change == 'partial': rows.pop()
                    if change == 'duplicate': rows.append(rows[0])
                    if change == 'depth': rows[0]['executed_depths'] = [0, 0]
                    if change == 'foreign': rows[0]['protocol_sha256'] = 'wrong'
                    (directory/'results.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        load_anchors(directory, limit=2, seed=42)


if __name__ == '__main__': unittest.main()
