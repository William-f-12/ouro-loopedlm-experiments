"""Optional, isolated RQ2 diagnostics over completed baseline trajectories."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-rq2', action='store_true', help='Off by default; disabled means no reads/model/output')
    parser.add_argument('--source-results', type=Path, help='Completed baseline directory or downloaded result ZIP')
    parser.add_argument('--output-dir', type=Path, help='New diagnostic directory, separate from baseline results')
    parser.add_argument('--dataset', choices=['math500', 'gsm8k'])
    parser.add_argument('--limit', type=int, default=20, help='Fixed seeded problem subset, keeping both arms/all thresholds')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-probe-tokens', type=int, default=0, help='0 = all saved tokens; positive = explicitly censored prefix')
    parser.add_argument('--scratch-dir', type=Path, help='Local fast disk for temporary BF16 reference tensors')
    parser.add_argument('--allow-cpu-test', action='store_true', help='Invalid scientific data; tiny offline tests only')
    args = parser.parse_args(argv)
    if args.enable_rq2 and (args.source_results is None or args.output_dir is None or args.limit < 1 or args.max_probe_tokens < 0):
        parser.error('Enabled probes require source/output, positive limit and nonnegative token cap')
    return args


def _verify_sources(protocol):
    hashes = protocol.get('source_sha256', {})
    required = {'ouro_local/modeling_ouro.py', 'ouro_local/configuration_ouro.py',
                'ouro_local/config.json', 'experiments/generation.py', 'experiments/quantization.py',
                'benchmark.py', 'upstream.json'}
    if not required <= set(hashes):
        raise ValueError('Source manifest lacks baseline model/generation/quantization provenance')
    for name, expected in hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Unrecognized baseline source path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Baseline source differs from this probe copy: {name}')


def _diagnostic(step, row, precision):
    from experiments.rq2 import stability_metrics
    index, exit_depth = step['token_index'], step['candidate_exit_depth']
    top = step['top1_by_loop']
    if precision == row['arm'] and (exit_depth != row['selected_depths'][index] or top[exit_depth - 1] != row['token_ids'][index]):
        raise ValueError(f"Anchor replay mismatch: {row['dataset']}:{row['example_id']} {precision} threshold={row['threshold']} token={index}. Check model/source/runtime/cache policy; no scientific summary accepted.")
    return {k: v for k, v in step.items() if k not in ('hidden', 'token_index')} | stability_metrics(top, exit_depth)


def run(args):
    if not args.enable_rq2:
        print('RQ2 disabled: no model loaded, no input read, no output written.')
        return None
    from experiments.rq2_io import load_anchors, save_probe_summary
    from experiments.reporting import write_json
    output = args.output_dir.resolve()
    origin = args.source_results.resolve()
    if output == origin or (origin.is_dir() and output.is_relative_to(origin)):
        raise ValueError('Probe output must be outside the baseline input directory')
    scratch_root = args.scratch_dir.resolve() if args.scratch_dir else Path(tempfile.gettempdir()).resolve()
    if scratch_root == origin or (origin.is_dir() and scratch_root.is_relative_to(origin)):
        raise ValueError('Probe scratch directory must be outside the read-only baseline input')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Probe output already exists and is not empty; choose a new directory')
    protocol, anchors, source = load_anchors(args.source_results, limit=args.limit, seed=args.seed, dataset=args.dataset)
    _verify_sources(protocol)
    if source['quantization'].get('format') != 'NVFP4_E2M1_E4M3FN_RTN':
        raise ValueError('Unsupported source quantization recipe')
    import torch
    import transformers
    from transformers import AutoTokenizer, set_seed
    from ouro_local.configuration_ouro import OuroConfig
    from ouro_local.modeling_ouro import OuroForCausalLM
    from experiments.quantization import quantize_core_mlp_
    from experiments.rq2 import replay_steps, hidden_errors
    if transformers.__version__ != '4.57.1':
        raise RuntimeError('Install the fixed requirements-bench.txt versions')
    if args.allow_cpu_test:
        if protocol.get('research_valid'):
            raise ValueError('Cannot replay scientific BF16 results as a CPU float32 test')
        device, dtype = 'cpu', torch.float32
        torch.set_num_threads(2)
    else:
        if not protocol.get('research_valid') or protocol['runtime']['dtype'] != 'bfloat16':
            raise ValueError('Research probes require valid BF16 baseline results')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError('Native BF16 CUDA GPU required; T4 is not supported')
        device, dtype = 'cuda', torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(protocol['seed'])
    config = OuroConfig.from_dict(protocol['config'])
    tokenizer = AutoTokenizer.from_pretrained(protocol['model_id'], revision=protocol['revision'], trust_remote_code=False)
    encoded = []
    for row in anchors:
        ids = tokenizer(row['prompt'], add_special_tokens=False, return_tensors='pt', return_token_type_ids=False).input_ids
        if ids.shape[1] != row['input_tokens']:
            raise ValueError('Tokenized prompt length differs from saved baseline')
        n = min(len(row['token_ids']), args.max_probe_tokens or len(row['token_ids']))
        if ids.shape[1] + n > min(config.max_position_embeddings, protocol['max_total_tokens']):
            raise ValueError('Replay exceeds the source context budget')
        encoded.append((row, ids, n))
    runtime = {'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__,
               'device': torch.cuda.get_device_name() if device == 'cuda' else 'CPU', 'dtype': str(dtype),
               'cuda': torch.version.cuda, 'attention': protocol['runtime']['attention'], 'tf32': False}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'research_valid': not args.allow_cpu_test, 'source': source, 'source_protocol': protocol, 'runtime': runtime,
        'selection': {'seed': args.seed, 'limit': args.limit, 'dataset': args.dataset},
        'max_probe_tokens': args.max_probe_tokens, 'probe_tokens': sum(n for _, _, n in encoded),
        'censored_anchors': sum(n < len(r['token_ids']) for r, _, n in encoded),
        'hidden_state_location': 'post-core-RMSNorm, before exit gate / LM head',
        'replay_policy': 'same recorded prompt/tokens/executed-depth schedule in both precisions; full current-token diagnostics, then restore anchor last-exit KV',
        'counterfactual_policy': 'candidate gate exits observed only; never substituted into recorded history',
        'timing_scope': 'diagnostic wall time, not baseline latency/throughput; no new free generation',
        'code_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__), ROOT/'experiments/rq2.py', ROOT/'experiments/rq2_io.py']}}
    write_json(output/'rq2_manifest.json', manifest)
    started = time.perf_counter()
    try:
        model = OuroForCausalLM.from_pretrained(protocol['model_id'], revision=protocol['revision'],
            config=config, dtype=dtype, device_map={'': device}, attn_implementation=runtime['attention']).eval()
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='ouro-rq2-', dir=scratch_root) as tmp:
            scratch = Path(tmp).resolve()
            for index, (row, ids, n) in enumerate(encoded):
                steps, hidden = [], []
                for step in replay_steps(model, ids.to(device), row['token_ids'][:n], row['executed_depths'][:n],
                                         threshold=row['threshold'], execution_mode=protocol.get('execution_mode', 'simulated')):
                    steps.append(_diagnostic(step, row, 'bf16'))
                    hidden.append(step['hidden'])
                torch.save({'hidden': torch.stack(hidden), 'steps': steps}, scratch/f'{index:05d}.pt')
                del hidden, steps, step
                print(f'BF16 reference {index+1}/{len(encoded)}: {row["dataset"]}:{row["example_id"]} anchor={row["arm"]} threshold={row["threshold"]} tokens={n}', flush=True)
            quantization = quantize_core_mlp_(model)
            write_json(output/'rq2_quantization.json', quantization)
            with (output/'rq2_tokens.jsonl').open('w', encoding='utf-8') as stream:
                for index, (row, ids, n) in enumerate(encoded):
                    reference = torch.load(scratch/f'{index:05d}.pt', map_location='cpu', weights_only=True)
                    for step in replay_steps(model, ids.to(device), row['token_ids'][:n], row['executed_depths'][:n],
                                             threshold=row['threshold'], execution_mode=protocol.get('execution_mode', 'simulated')):
                        pos = step['token_index']
                        record = {'dataset': row['dataset'], 'example_id': row['example_id'], 'anchor_arm': row['arm'],
                            'threshold': row['threshold'], 'execution_mode': protocol.get('execution_mode', 'simulated'),
                            'anchor_correct': row['correct'], 'anchor_generated_tokens': len(row['token_ids']),
                            'token_index': pos, 'token_position_fraction': (pos + 1) / len(row['token_ids']),
                            'token_id': row['token_ids'][pos],
                            'token_text': tokenizer.decode([row['token_ids'][pos]], skip_special_tokens=False),
                            'recorded_selected_depth': row['selected_depths'][pos],
                            'history_executed_depth': row['executed_depths'][pos],
                            'bf16': reference['steps'][pos], 'ptq_nvfp4': _diagnostic(step, row, 'ptq_nvfp4'),
                            'hidden_error': hidden_errors(reference['hidden'][pos], step['hidden'])}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    del reference, step
                    print(f'PTQ paired probe {index+1}/{len(encoded)} complete', flush=True)
        summary = save_probe_summary(output)
        manifest.update(status='complete', diagnostic_wall_seconds=time.perf_counter()-started,
                        anchor_replay_check='all observed own-precision token IDs and selected depths matched')
        write_json(output/'rq2_manifest.json', manifest)
        print(f'RQ2 complete: {output}', flush=True)
        return summary
    except Exception as error:
        manifest.update(status='failed', error=str(error), diagnostic_wall_seconds=time.perf_counter()-started)
        write_json(output/'rq2_manifest.json', manifest)
        raise


def main():
    try:
        run(parse_args())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f'RQ2 failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
