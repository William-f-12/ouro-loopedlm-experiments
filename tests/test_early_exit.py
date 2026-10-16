"""Physical loop stopping and last-exit cache fill, independent of trace labels."""
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import torch
from transformers import DynamicCache
from ouro_local.modeling_ouro import UniversalTransformerCache
from experiments.generation import generate_traced, select_depth
from experiments.reporting import summarize
from test_baselines import tiny_model


def kv(cache, index):
    if isinstance(cache, UniversalTransformerCache):
        return cache.key_cache[index], cache.value_cache[index]
    return cache.layers[index].keys, cache.layers[index].values


class EarlyExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def gate_at(self, model, depth):
        calls = []
        def gate(_module, _args, output):
            calls.append(1)
            return torch.full_like(output, 100 if len(calls) == depth else -100)
        return model.model.early_exit_gate.register_forward_hook(gate), calls

    def test_changing_depth_matches_independent_cache_fill_oracle(self):
        for cache_class in (DynamicCache, UniversalTransformerCache):
            with self.subTest(cache=cache_class.__name__), torch.inference_mode():
                actual, oracle = tiny_model(), tiny_model()
                ac, oc = cache_class(), cache_class()
                # Full-depth prompt context is shared initially.
                prefix = torch.tensor([[4, 5]])
                actual(prefix, past_key_values=ac, exit_at_step=3, use_cache=True)
                oracle(prefix, past_key_values=oc, exit_at_step=3, use_cache=True)
                for offset, depth in enumerate((1, 4, 2, 3)):
                    before = [(k.clone(), v.clone()) for k, v in (kv(ac, i) for i in range(8))]
                    handle, gates = self.gate_at(actual, depth)
                    layer_calls = []
                    hooks = [layer.register_forward_hook(lambda *args: layer_calls.append(1))
                             for layer in actual.model.layers]
                    try:
                        result = actual(torch.tensor([[6 + offset]]), past_key_values=ac,
                            use_cache=True, exit_threshold=.5, true_early_exit=True)
                    finally:
                        handle.remove()
                        for hook in hooks:
                            hook.remove()
                    expected = oracle(torch.tensor([[6 + offset]]), past_key_values=oc,
                        use_cache=True, exit_at_step=depth - 1)
                    self.assertEqual(len(gates), depth)
                    self.assertEqual(len(layer_calls), depth * 2)
                    torch.testing.assert_close(result.logits, expected.logits)
                    # Oracle ran all loops. Replace only THIS token's skipped KV,
                    # without calling the production cache-fill implementation.
                    for loop in range(depth, 4):
                        for layer in range(2):
                            source_k, source_v = kv(oc, (depth - 1) * 2 + layer)
                            target_k, target_v = kv(oc, loop * 2 + layer)
                            target_k[:, :, -1:] = source_k[:, :, -1:]
                            target_v[:, :, -1:] = source_v[:, :, -1:]
                    for i in range(8):
                        ak, av = kv(ac, i)
                        ok, ov = kv(oc, i)
                        self.assertEqual(ac.get_seq_length(i), 3 + offset)
                        torch.testing.assert_close(ak, ok)
                        torch.testing.assert_close(av, ov)
                        torch.testing.assert_close(ak[:, :, :-1], before[i][0], rtol=0, atol=0)
                        torch.testing.assert_close(av[:, :, :-1], before[i][1], rtol=0, atol=0)

    def test_true_exit_uses_cumulative_conditional_hazards(self):
        model = tiny_model()
        values = iter((-math.log(3), 0., 0., 0.))
        calls = []
        def gate(_module, _args, output):
            calls.append(1)
            return torch.full_like(output, next(values))
        handle = model.model.early_exit_gate.register_forward_hook(gate)
        try:
            with torch.inference_mode():
                model(torch.tensor([[4]]), past_key_values=DynamicCache(),
                      use_cache=True, exit_threshold=.5, true_early_exit=True)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)  # PDF [.25, .375], cumulative .625
        depth, pdf = select_depth([torch.tensor([[[-math.log(3)]]]), torch.zeros(1, 1, 1)],
                                  .5, total_loops=4)
        self.assertEqual(depth, 2)
        torch.testing.assert_close(pdf, torch.tensor([.25, .375]))

    def test_real_decode_counts_physical_work_and_keeps_fixed_reference(self):
        model = tiny_model().to(torch.bfloat16)
        with torch.no_grad():
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(100)
        for ids in (torch.tensor([[4, 5, 6]]), torch.tensor([[4]])):
            counts = []
            hook = model.model.layers[0].register_forward_pre_hook(
                lambda _module, args: counts.append(args[0].shape[1]))
            try:
                adaptive = generate_traced(model, ids, threshold=.5, max_new_tokens=3,
                    eos_ids=[], execution_mode="early_exit")
            finally:
                hook.remove()
            self.assertEqual(adaptive["selected_depths"], [1] * 3)
            self.assertEqual(adaptive["executed_depths"], [1] * 3)
            self.assertEqual(adaptive["exit_pdfs"], [[1., None, None, None]] * 3)
            self.assertEqual(adaptive["actual_forward_token_loops"], (ids.shape[1] - 1) * 4 + 3)
            self.assertEqual(sum(counts), adaptive["actual_forward_token_loops"])
            fixed = generate_traced(model, ids, threshold=1., max_new_tokens=3,
                eos_ids=[], execution_mode="early_exit")
            old_fixed = generate_traced(model, ids, threshold=1., max_new_tokens=3, eos_ids=[])
            self.assertEqual(fixed["token_ids"], old_fixed["token_ids"])
            self.assertEqual(fixed["executed_depths"], [4] * 3)
            self.assertEqual(fixed["actual_forward_token_loops"], (ids.shape[1] + 2) * 4)
            eos = generate_traced(model, ids, threshold=.5, max_new_tokens=3,
                eos_ids=[adaptive["token_ids"][0]], execution_mode="early_exit")
            self.assertEqual(eos["generated_tokens"], 1)
            self.assertFalse(eos["truncated"])
        self.assertEqual(len(model.model._forward_hooks), 0)

    def test_invalid_true_forward_fails_before_cache_mutation(self):
        model = tiny_model()
        for ids, options in (
            (torch.tensor([[4, 5]]), {}),
            (torch.tensor([[4], [5]]), {}),
            (torch.tensor([[4]]), {"use_cache": False}),
            (torch.tensor([[4]]), {"labels": torch.tensor([[4]])}),
            (torch.tensor([[4]]), {"exit_at_step": 2}),
            (torch.tensor([[4]]), {"use_weighted_exit": True}),
            (torch.tensor([[4]]), {"exit_threshold": 0.}),
        ):
            cache = DynamicCache()
            kwargs = dict(use_cache=True, exit_threshold=.5, true_early_exit=True)
            kwargs.update(options)
            with self.subTest(options=options), torch.inference_mode():
                with self.assertRaises((ValueError, TypeError)):
                    model(ids, past_key_values=cache, **kwargs)
            self.assertEqual(cache.get_seq_length(), 0)

    def test_trajectories_start_with_fresh_caches_and_report_actual_mode(self):
        model = tiny_model()
        ids = torch.tensor([[4, 5, 6]])
        first = generate_traced(model, ids, threshold=.5, max_new_tokens=4,
                                eos_ids=[], execution_mode="early_exit")
        generate_traced(model, ids, threshold=1., max_new_tokens=4,
                        eos_ids=[], execution_mode="early_exit")
        again = generate_traced(model, ids, threshold=.5, max_new_tokens=4,
                                eos_ids=[], execution_mode="early_exit")
        for field in ("token_ids", "executed_depths", "exit_pdfs"):
            self.assertEqual(first[field], again[field])
        row = dict(first, dataset="gsm8k", example_id="0", arm="bf16", threshold=.5,
                   correct=True, parse_success=True)
        report = summarize([row])
        self.assertEqual(report["execution_mode"], "early_exit")
        self.assertIn("last-exit", report["depth_semantics"])
        with self.assertRaisesRegex(ValueError, "execution mode"):
            summarize([row, dict(row, example_id="1", execution_mode="simulated")])

    def test_fallback_limit_other_thresholds_and_bf16_cdf(self):
        model = tiny_model().to(torch.bfloat16)
        for loops in (1, 3, 4):
            model.model.total_ut_steps = loops
            model.config.total_ut_steps = loops
            for hazard in (.0, .25, .5, .9, 1.):
                value = -100 if hazard == 0 else 100 if hazard == 1 else math.log(hazard / (1 - hazard))
                with torch.no_grad():
                    model.model.early_exit_gate.weight.zero_()
                    model.model.early_exit_gate.bias.fill_(value)
                gates = [torch.full((1, 1, 1), value, dtype=torch.bfloat16)] * loops
                for threshold in (.1, .5, .9, .999, 1.):
                    expected, _pdf = select_depth(gates, threshold)
                    result = generate_traced(model, torch.tensor([[4]]), threshold=threshold,
                        max_new_tokens=2, eos_ids=[], execution_mode="early_exit")
                    self.assertEqual(result["executed_depths"], [expected, expected])

    def test_unsupported_cache_training_and_error_cleanup(self):
        from transformers import StaticCache
        model = tiny_model()
        ids = torch.tensor([[4]])
        with torch.inference_mode():
            with self.assertRaisesRegex(TypeError, "DynamicCache"):
                model(ids, use_cache=True, true_early_exit=True, exit_threshold=.5,
                      past_key_values=StaticCache(config=model.config, max_cache_len=16))
            model.train()
            with self.assertRaisesRegex(ValueError, "eval"):
                model(ids, use_cache=True, true_early_exit=True, exit_threshold=.5)
        model.eval()
        original = (model.early_exit_step, model.early_exit_threshold)
        def fail(_module, _args):
            raise RuntimeError("injected failure")
        handle = model.model.layers[0].register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                generate_traced(model, ids, threshold=.5, max_new_tokens=2,
                                eos_ids=[], execution_mode="early_exit")
        finally:
            handle.remove()
        self.assertEqual(len(model.model._forward_hooks), 0)
        self.assertEqual((model.early_exit_step, model.early_exit_threshold), original)


if __name__ == "__main__":
    unittest.main()
