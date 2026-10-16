"""Independent batch-one greedy trajectories with simulated or physical exit."""
import math
import time
import torch
from transformers import DynamicCache


def select_depth(gate_list, threshold, *, total_loops=None):
    """Use upstream conditional hazards in their original arithmetic dtype.

    At threshold=1 use the final loop explicitly, avoiding BF16 CDF rounding
    accidentally reaching 1 before the final loop. Return a one-based depth.
    """
    if not math.isfinite(threshold) or not 0 < threshold <= 1 or not gate_list:
        raise ValueError("Expected exit gates and a threshold in (0, 1]")
    total_loops = len(gate_list) if total_loops is None else total_loops
    if total_loops < len(gate_list):
        raise ValueError("Observed more loops than configured")
    gates = [gate[:, -1, 0] for gate in gate_list]
    if gates[0].numel() != 1:
        raise ValueError("Depth instrumentation supports batch size 1 only")
    remaining = torch.ones_like(gates[0])
    pdf = []
    for i, gate in enumerate(gates):
        hazard = torch.sigmoid(gate)
        pdf.append(remaining if i == total_loops - 1 else hazard * remaining)
        remaining = remaining * (1 - hazard)
    pdf = torch.stack(pdf, dim=-1).flatten()
    if not torch.isfinite(pdf).all():
        raise ValueError("Non-finite exit distribution")
    crossing = torch.nonzero(pdf.cumsum(-1) >= threshold).flatten()
    if len(gates) < total_loops and (threshold == 1 or not len(crossing)):
        raise ValueError("Execution stopped before the threshold crossing")
    depth = len(gates) if threshold == 1 or not len(crossing) else crossing[0].item() + 1
    return depth, pdf


@torch.inference_mode()
def generate_traced(model, input_ids, *, threshold, max_new_tokens, eos_ids, execution_mode="simulated"):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("Expected a nonempty, unpadded batch of one prompt")
    if max_new_tokens < 1 or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Invalid token limit or exit threshold")
    if execution_mode not in ("simulated", "early_exit"):
        raise ValueError("execution_mode must be simulated or early_exit")
    if model.training:
        raise ValueError("Generation requires model.eval()")
    device = input_ids.device
    loops = model.model.total_ut_steps
    adaptive = execution_mode == "early_exit" and threshold < 1
    captured = {}
    recording = False

    def record_core(_module, _args, output):
        if not recording:
            return
        # Do not retain prompt hidden states or the complete cache in the trace.
        gate_list = output[2]
        depth, pdf = select_depth(gate_list, threshold, total_loops=loops)
        if adaptive and depth != len(gate_list):
            raise RuntimeError("Actual loop execution disagrees with threshold crossing")
        values = pdf.float().tolist() + [None] * (loops - len(gate_list))
        captured.update(depth=depth, pdf=values, executed=len(gate_list))

    handle = model.model.register_forward_hook(record_core)
    cache = DynamicCache()
    tokens, depths, executed, pdfs = [], [], [], []
    forward_token_loops = 0
    old_threshold, old_step = model.early_exit_threshold, model.early_exit_step
    model.early_exit_threshold, model.early_exit_step = threshold, None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    stopped = False
    try:
        next_input = input_ids
        if adaptive:
            if input_ids.shape[1] > 1:
                # Full-depth context, then make the FIRST generation decision
                # with one adaptive query token, just like subsequent decisions.
                prefix = input_ids[:, :-1]
                output = model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                               past_key_values=cache, use_cache=True, logits_to_keep=1,
                               exit_at_step=loops - 1)
                cache = output.past_key_values
                forward_token_loops += prefix.shape[1] * loops
                del output
            next_input = input_ids[:, -1:]
        recording = True
        for _ in range(max_new_tokens):
            captured.clear()
            mask = torch.ones((1, input_ids.shape[1] + len(tokens)), dtype=torch.long, device=device)
            options = {"exit_at_step": loops - 1} if threshold == 1 else {"true_early_exit": adaptive}
            output = model(input_ids=next_input, attention_mask=mask,
                           past_key_values=cache, use_cache=True, logits_to_keep=1, **options)
            if not captured or not torch.isfinite(output.logits[:, -1]).all():
                raise RuntimeError("Missing loop trace or non-finite logits")
            cache = output.past_key_values
            token = output.logits[0, -1].argmax().item()
            tokens.append(token)
            depths.append(captured["depth"])
            executed.append(captured["executed"])
            pdfs.append(captured["pdf"])
            forward_token_loops += next_input.shape[1] * captured["executed"]
            if token in eos_ids:
                stopped = True
                break
            next_input = input_ids.new_tensor([[token]])
    finally:
        handle.remove()
        model.early_exit_threshold, model.early_exit_step = old_threshold, old_step
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "execution_mode": execution_mode,
        "token_ids": tokens, "generated_tokens": len(tokens), "input_tokens": input_ids.shape[1],
        "selected_depths": depths, "executed_depths": executed, "exit_pdfs": pdfs,
        "selected_core_steps": sum(depths), "executed_core_steps": sum(executed),
        "mean_selected_depth": sum(depths) / len(depths),
        "mean_executed_depth": sum(executed) / len(executed),
        "actual_forward_token_loops": forward_token_loops,
        "stop_reason": "eos" if stopped else "length", "truncated": not stopped,
        "generation_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
    }
