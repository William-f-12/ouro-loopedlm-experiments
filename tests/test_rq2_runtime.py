"""Real offline saved-baseline -> probe CLI, without Hugging Face downloads."""
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
from test_baselines import tiny_model


class RuntimeTests(unittest.TestCase):
    def test_saved_baseline_replayed_in_both_precisions_without_source_mutation(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)/'probe project'
            project.mkdir()
            for name in ('ouro_local', 'experiments', 'scripts'):
                shutil.copytree(ROOT/name, project/name, ignore=shutil.ignore_patterns('__pycache__'))
            for name in ('benchmark.py', 'inference.py', 'requirements.txt', 'requirements-bench.txt'):
                shutil.copyfile(ROOT/name, project/name)
            model = tiny_model()
            with torch.no_grad():
                model.model.early_exit_gate.weight.zero_()
                model.model.early_exit_gate.bias.fill_(2.)
            weights = Path(tmp)/'tiny weights'
            model.save_pretrained(weights)
            model.config.save_pretrained(project/'ouro_local')
            words = ['<eos>', '<unk>'] + [str(i) for i in range(30)]
            token = Tokenizer(WordLevel(dict(zip(words, range(32))), unk_token='<unk>'))
            token.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=token, eos_token='<eos>', unk_token='<unk>')
            tokenizer.chat_template = "{% for m in messages %}{{m['content']}}{% endfor %}"
            tokenizer.save_pretrained(weights)
            (project/'upstream.json').write_text(json.dumps({'model_id': str(weights), 'revision': 'offline-tiny'}))
            data = project/'tiny.jsonl'
            data.write_text(json.dumps({'id': '0', 'dataset': 'gsm8k', 'question': '1 + 1?', 'answer': '#### 2'})+'\n')
            env = os.environ | {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONUTF8': '1'}
            def execute(command):
                return subprocess.run([sys.executable, '-B'] + list(map(str, command)), cwd=project,
                                      env=env, capture_output=True, text=True, encoding='utf-8', timeout=180)
            for mode in ('early_exit', 'simulated'):
                baseline = project/f'baseline-{mode}'
                result = execute([project/'benchmark.py', '--allow-cpu-test', '--datasets', 'gsm8k',
                    '--data-file', data, '--limit', '1', '--max-new-tokens', '3', '--max-total-tokens', '128',
                    '--execution-mode', mode, '--output-dir', baseline])
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                before = {p.name: p.read_bytes() for p in baseline.iterdir() if p.is_file()}
                out = project/f'probe-{mode}'
                command = [project/'scripts/rq2_probe.py', '--enable-rq2', '--allow-cpu-test',
                           '--source-results', baseline, '--output-dir', out, '--limit', '1']
                result = execute(command)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertEqual(before, {p.name:p.read_bytes() for p in baseline.iterdir() if p.is_file()})
                manifest = json.loads((out/'rq2_manifest.json').read_text())
                self.assertEqual(manifest['status'], 'complete')
                self.assertFalse(manifest['research_valid'])
                records = [json.loads(x) for x in (out/'rq2_tokens.jsonl').read_text().splitlines()]
                self.assertEqual(len(records), sum(json.loads(x)['generated_tokens'] for x in before['results.jsonl'].splitlines()))
                self.assertEqual({r['execution_mode'] for r in records}, {mode})
                self.assertTrue(all(len(r['hidden_error']['l2_by_loop']) == 4 for r in records))
                self.assertTrue(any(sum(r['hidden_error']['l2_by_loop']) > 0 for r in records))
                self.assertEqual(len(json.loads((out/'rq2_summary.json').read_text())['groups']), 4)
                self.assertNotEqual(execute(command).returncode, 0)  # No accidental overwrite.


if __name__ == '__main__': unittest.main()
