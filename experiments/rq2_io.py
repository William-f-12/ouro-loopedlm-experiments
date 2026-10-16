"""Read-only baseline input validation and problem-level RQ2 summaries."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import zipfile
from .data import digest_json
from .reporting import mean_se, record_key, write_json


def load_anchors(source, *, limit=20, seed=42, dataset=None):
    source = Path(source).resolve()
    if limit < 1:
        raise ValueError('Probe problem limit must be positive')
    names = ('manifest.json', 'results.jsonl', 'quantization.json')
    if source.is_dir():
        payload = {name: (source / name).read_bytes() for name in names}
    else:
        with zipfile.ZipFile(source) as z:
            if len(z.namelist()) != len(set(z.namelist())):
                raise ValueError('Duplicate entries in baseline result ZIP')
            payload = {name: z.read(name) for name in names}
    manifest = json.loads(payload['manifest.json'])
    protocol, identity = manifest['protocol'], manifest['protocol_sha256']
    if digest_json(protocol) != identity:
        raise ValueError('Baseline manifest protocol hash mismatch')
    mode = protocol.get('execution_mode', 'simulated')
    if mode not in ('early_exit', 'simulated') or protocol.get('decoding') != 'greedy' or protocol.get('batch_size') != 1:
        raise ValueError('Only greedy batch-one baseline trajectories are supported')
    if set(protocol['arms']) != {'bf16', 'ptq_nvfp4'}:
        raise ValueError('Expected paired BF16 and PTQ baseline arms')
    loops = protocol['config']['total_ut_steps']
    thresholds = protocol['thresholds']
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not math.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError('Invalid source thresholds')
    examples = [(x['dataset'], str(x['id'])) for x in protocol['examples']]
    if not examples or len(set(examples)) != len(examples):
        raise ValueError('Duplicate or empty source examples')
    expected = {(arm, ds, ident, t) for ds, ident in examples for arm in protocol['arms'] for t in thresholds}
    rows = [json.loads(line) for line in payload['results.jsonl'].splitlines() if line.strip()]
    seen = set()
    for row in rows:
        key = record_key(row)
        if key in seen or key not in expected or row.get('protocol_sha256') != identity:
            raise ValueError('Duplicate, unexpected or foreign-protocol baseline record')
        seen.add(key)
        if row.get('execution_mode', 'simulated') != mode:
            raise ValueError('Mixed execution modes in baseline results')
        tokens, selected, executed = (row[k] for k in ('token_ids', 'selected_depths', 'executed_depths'))
        n = row['generated_tokens']
        if not tokens or not len(tokens) == len(selected) == len(executed) == n:
            raise ValueError('Token/depth trace length mismatch')
        if any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('Invalid saved token ID')
        if any(type(d) is not int or not 1 <= d <= loops for d in selected + executed):
            raise ValueError('Invalid recorded exit depth')
        adaptive = mode == 'early_exit' and row['threshold'] < 1
        if (adaptive and selected != executed) or (not adaptive and any(d != loops for d in executed)):
            raise ValueError('Saved depth trace contradicts its execution policy')
        if row['threshold'] == 1 and any(d != loops for d in selected):
            raise ValueError('Threshold-one reference must select final loop')
        if not isinstance(row['prompt'], str) or not row['prompt'] or row['input_tokens'] < 1:
            raise ValueError('Invalid saved prompt')
    if seen != expected:
        raise ValueError('Baseline batch is incomplete; finish it before probing')
    eligible = sorted(k for k in examples if dataset is None or k[0] == dataset)
    if not eligible:
        raise ValueError('No baseline examples match the selected dataset')
    random.Random(seed).shuffle(eligible)
    chosen = eligible[:limit]
    ordering = {key: i for i, key in enumerate(chosen)}
    anchors = [r for r in rows if (r['dataset'], str(r['example_id'])) in ordering]
    anchors.sort(key=lambda r: (ordering[r['dataset'], str(r['example_id'])], r['threshold'], r['arm']))
    provenance = {'source_path': str(source), 'protocol_sha256': identity,
                  'file_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
                  'selected_examples': chosen, 'quantization': json.loads(payload['quantization.json'])}
    return protocol, anchors, provenance


def summarize_probes(rows):
    # Accumulate per-problem means first: token count is NOT the independent
    # sample count for SE. Keep anchor precision, mode and threshold separate.
    totals = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0., 0])))
    counts = defaultdict(int)
    for row in rows:
        key = (row['dataset'], row['execution_mode'], row['threshold'], row['anchor_arm'])
        values = {}
        for arm in ('bf16', 'ptq_nvfp4'):
            info = row[arm]
            values.update({f'{arm}_candidate_depth': info['candidate_exit_depth'],
                f'{arm}_stable_loop': info['stable_from_loop'],
                f'{arm}_late_exit_rate': float(info['exit_minus_stability'] > 0),
                f'{arm}_early_exit_rate': float(info['exit_minus_stability'] < 0),
                f'{arm}_extra_loops_after_stability': info['extra_loops_after_stability']})
        values['candidate_depth_delta_ptq_minus_bf16'] = row['ptq_nvfp4']['candidate_exit_depth'] - row['bf16']['candidate_exit_depth']
        values['stable_loop_delta_ptq_minus_bf16'] = row['ptq_nvfp4']['stable_from_loop'] - row['bf16']['stable_from_loop']
        for label in ('l2_by_loop', 'relative_l2_by_loop'):
            for loop, value in enumerate(row['hidden_error'][label], 1):
                values[f'{label}_{loop}'] = value
        counts[key] += 1
        for label, value in values.items():
            if value is not None:
                pair = totals[key][row['example_id']][label]
                pair[0] += value
                pair[1] += 1
    groups = []
    for key, problems in sorted(totals.items()):
        labels = sorted({label for metrics in problems.values() for label in metrics})
        metrics = {label: mean_se(v[label][0] / v[label][1] for v in problems.values() if label in v) for label in labels}
        groups.append(dict(zip(('dataset', 'execution_mode', 'threshold', 'anchor_arm'), key)) |
                      {'examples': len(problems), 'token_decisions': counts[key], 'metrics': metrics})
    return {'schema_version': 1, 'aggregation': 'equal-weight problem means; SE across problems, not tokens',
            'interpretation': 'Cross-precision candidate exits are conditional on shared recorded tokens/depth history. Full-loop stability is diagnostic, not correctness or actual executed work.',
            'groups': groups}


def save_probe_summary(directory):
    directory = Path(directory)
    with (directory / 'rq2_tokens.jsonl').open(encoding='utf-8') as stream:
        summary = summarize_probes(json.loads(line) for line in stream if line.strip())
    write_json(directory / 'rq2_summary.json', summary)
    flat = []
    for group in summary['groups']:
        row = {k: v for k, v in group.items() if k != 'metrics'}
        for name, metric in group['metrics'].items():
            row[name], row[name + '_se'], row[name + '_n'] = metric['mean'], metric['se'], metric['n']
        flat.append(row)
    with (directory / 'rq2_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        writer.writerows(flat)
    return summary
