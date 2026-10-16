"""Full-loop diagnostics on fixed token prefixes and recorded cache histories.

Never called by the baseline generator. Diagnostic continuation is discarded
from skipped-loop KV before the next token, preserving the anchor's history.
"""
import math
import torch
from transformers import DynamicCache
from .generation import select_depth


def stability_metrics(top1, exit_depth):
    if not top1 or not 1 <= exit_depth <= len(top1):
        raise ValueError('Expected loop predictions and a valid one-based exit depth')
    stable = len(top1)
    while stable > 1 and top1[stable - 2] == top1[-1]:
        stable -= 1
    gap = exit_depth - stable
    return {'stable_from_loop': stable, 'exit_minus_stability': gap,
            'extra_loops_after_stability': max(gap, 0),
            'exit_before_stability': gap < 0,
            'exit_top1_matches_final_loop': top1[exit_depth - 1] == top1[-1]}


def hidden_errors(reference, quantized):
    if reference.ndim != 2 or reference.shape != quantized.shape:
        raise ValueError('Hidden states must have identical [loops, hidden] shape')
    a, b = reference.float(), quantized.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Non-finite hidden state')
    norms = torch.linalg.vector_norm(a, dim=-1)
    errors = torch.linalg.vector_norm(b - a, dim=-1)
    return {'l2_by_loop': errors.tolist(),
            'reference_norm_by_loop': norms.tolist(),
            'relative_l2_by_loop': [e / n if n > 0 else None for e, n in zip(errors.tolist(), norms.tolist())]}


def _restore_anchor_cache(cache, depth, loops, layers):
    # Full diagnostic forward appended a real KV at EVERY loop. Replace only
    # this position in skipped loops; all earlier positions stay untouched.
    for layer in range(layers):
        source = cache.layers[(depth - 1) * layers + layer]
        for loop in range(depth, loops):
            target = cache.layers[loop * layers + layer]
            if target.keys.shape != source.keys.shape:
                raise ValueError('Diagnostic caches lost token alignment')
            target.keys[:, :, -1:].copy_(source.keys[:, :, -1:])
            target.values[:, :, -1:].copy_(source.values[:, :, -1:])


def replay_steps(model, input_ids, token_ids, depths, *, threshold, execution_mode):
    """Yield CPU hidden states and scalar diagnostics for every saved decision.

depths are the anchor's ACTUALLY EXECUTED depths, shared across precisions.
The candidate gate exit is observed, never used to change that shared history.
Input token i is prompt[-1] for i=0 and saved output[i-1] afterwards: the last
saved output (including EOS) is predicted but never forwarded again.
"""
    loops = model.model.total_ut_steps
    if model.training or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError('Replay requires eval mode and one nonempty prompt')
    if not math.isfinite(threshold) or not 0 < threshold <= 1 or execution_mode not in ('early_exit', 'simulated'):
        raise ValueError('Invalid replay threshold or execution mode')
    if not token_ids or len(token_ids) != len(depths) or any(type(d) is not int or not 1 <= d <= loops for d in depths):
        raise ValueError('Saved tokens and executed depths must be aligned and nonempty')
    if any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in token_ids):
        raise ValueError('Saved token ID outside vocabulary')
    adaptive = execution_mode == 'early_exit' and threshold < 1
    if not adaptive and any(d != loops for d in depths):
        raise ValueError('Simulated/fixed-depth histories must execute all loops')
    cache = DynamicCache()
    with torch.inference_mode():
        next_input = input_ids
        if adaptive:
            if input_ids.shape[1] > 1:
                prefix = input_ids[:, :-1]
                out, _, _ = model.model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                        past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                del out
            next_input = input_ids[:, -1:]
        for index, depth in enumerate(depths):
            mask = input_ids.new_ones((1, input_ids.shape[1] + index))
            out, hidden, gates = model.model(input_ids=next_input, attention_mask=mask,
                                             past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if len(hidden) != loops:
                raise RuntimeError('Diagnostic forward must observe every loop')
            candidate, pdf = select_depth(gates, threshold, total_loops=loops)
            # Same one-position LM-head shape used by baseline generation.
            top = []
            for state in hidden:
                logits = model.lm_head(state[:, -1:])
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite diagnostic logits')
                top.append(logits[0, -1].argmax().item())
            step = {'token_index': index, 'top1_by_loop': top,
                    'candidate_exit_depth': candidate, 'exit_pdf': pdf.float().tolist(),
                    'hidden': torch.stack([h[0, -1].detach().cpu() for h in hidden])}
            if adaptive:
                _restore_anchor_cache(cache, depth, loops, model.config.num_hidden_layers)
            del out, hidden, gates, logits
            yield step
            next_input = input_ids.new_tensor([[token_ids[index]]])
