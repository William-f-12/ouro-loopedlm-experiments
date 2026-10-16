"""RQ2 diagnostic invariants: stable suffix, matched prefixes, unmodified rollout."""
import importlib.util
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class AvailabilityTests(unittest.TestCase):
    def test_replay_module_available(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.rq2'),
                             'Optional RQ2 replay is not implemented')


if importlib.util.find_spec('experiments.rq2') is not None:
    import torch
    from test_baselines import tiny_model
    from experiments.generation import generate_traced
    from experiments.quantization import quantize_core_mlp_
    from experiments.rq2 import stability_metrics, hidden_errors, replay_steps

    class ProbeTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            torch.set_num_threads(2)

        def test_stability_means_constant_suffix_not_first_repeated_prediction(self):
            for top, depth, stable, gap in [
                ([7, 7, 7, 7], 4, 1, 3), ([7, 8, 8, 8], 1, 2, -1),
                ([7, 7, 8, 8], 3, 3, 0), ([7, 8, 7, 7], 4, 3, 1),
                ([7, 8, 7, 8], 2, 4, -2), ([9], 1, 1, 0),
            ]:
                result = stability_metrics(top, depth)
                self.assertEqual(result['stable_from_loop'], stable)
                self.assertEqual(result['exit_minus_stability'], gap)
            for top, depth in [([], 1), ([1, 2], 0), ([1, 2], 3)]:
                with self.assertRaises(ValueError): stability_metrics(top, depth)

        def test_hidden_error_known_norms_and_zero_denominator(self):
            ref = torch.tensor([[3., 4.], [0., 0.]])
            q = torch.tensor([[6., 8.], [1., 0.]])
            result = hidden_errors(ref, q)
            self.assertEqual(result['l2_by_loop'], [5., 1.])
            self.assertEqual(result['relative_l2_by_loop'], [1., None])
            self.assertEqual(hidden_errors(ref, ref)['l2_by_loop'], [0., 0.])

        def test_replay_matches_saved_tokens_depths_and_does_not_change_generation(self):
            for dtype in [torch.float32, torch.bfloat16]:
                for mode in ['early_exit', 'simulated']:
                    for threshold in [.5, 1.]:
                        for prompt in [[4], [4, 5, 6]]:
                            with self.subTest(dtype=dtype, mode=mode, threshold=threshold, prompt=prompt):
                                model = tiny_model().to(dtype)
                                with torch.no_grad():
                                    model.model.early_exit_gate.weight.zero_()
                                    model.model.early_exit_gate.bias.fill_(2.)
                                ids = torch.tensor([prompt])
                                kw = dict(threshold=threshold, max_new_tokens=5,
                                          eos_ids=[], execution_mode=mode)
                                baseline = generate_traced(model, ids, **kw)
                                probe = list(replay_steps(model, ids, baseline['token_ids'],
                                    baseline['executed_depths'], threshold=threshold, execution_mode=mode))
                                self.assertEqual(len(probe), 5)
                                for i, step in enumerate(probe):
                                    self.assertEqual(step['candidate_exit_depth'], baseline['selected_depths'][i])
                                    self.assertEqual(step['top1_by_loop'][step['candidate_exit_depth'] - 1], baseline['token_ids'][i])
                                    self.assertEqual(tuple(step['hidden'].shape), (4, 32))
                                    self.assertEqual(step['hidden'].device.type, 'cpu')
                                again = generate_traced(model, ids, **kw)
                                for key in ['token_ids', 'exit_pdfs', 'executed_depths']:
                                    self.assertEqual(baseline[key], again[key])
                                self.assertEqual(len(model.model._forward_hooks), 0)

        def test_same_anchor_zero_self_error_and_nonzero_ptq_error(self):
            model = tiny_model()
            ids = torch.tensor([[4, 5, 6]])
            tokens, depths = [7, 8, 9, 10], [1, 4, 2, 3]
            kw = dict(threshold=.5, execution_mode='early_exit')
            ref = list(replay_steps(model, ids, tokens, depths, **kw))
            repeat = list(replay_steps(model, ids, tokens, depths, **kw))
            for a, b in zip(ref, repeat):
                self.assertEqual(hidden_errors(a['hidden'], b['hidden'])['l2_by_loop'], [0.] * 4)
            quantize_core_mlp_(model)
            q = list(replay_steps(model, ids, tokens, depths, **kw))
            self.assertTrue(all(sum(hidden_errors(a['hidden'], b['hidden'])['l2_by_loop']) > 0 for a, b in zip(ref, q)))

        def test_variable_depth_history_matches_independent_physical_exit_states(self):
            from transformers import DynamicCache
            model = tiny_model()
            prompt = torch.tensor([[4, 5, 6]])
            tokens, depths = [7, 8, 9, 10], [1, 4, 2, 3]
            actual = []
            with torch.inference_mode():
                cache = DynamicCache()
                model.model(input_ids=prompt[:, :-1], attention_mask=torch.ones_like(prompt[:, :-1]),
                            past_key_values=cache, use_cache=True)
                for pos, depth in enumerate(depths):
                    called = []
                    def force_exit(_module, _args, output):
                        called.append(1)
                        return torch.full_like(output, 100 if len(called) >= depth else -100)
                    hook = model.model.early_exit_gate.register_forward_hook(force_exit)
                    token = prompt[:, -1:] if pos == 0 else torch.tensor([[tokens[pos-1]]])
                    try:
                        _, hidden, _ = model.model(input_ids=token, attention_mask=torch.ones(1, prompt.shape[1]+pos),
                            past_key_values=cache, use_cache=True, halt_threshold=.5)
                    finally:
                        hook.remove()
                    self.assertEqual(len(hidden), depth)
                    actual.append(torch.stack([h[0, -1].clone() for h in hidden]))
            replayed = list(replay_steps(model, prompt, tokens, depths, threshold=.5, execution_mode='early_exit'))
            for pos, depth in enumerate(depths):
                torch.testing.assert_close(replayed[pos]['hidden'][:depth], actual[pos], rtol=0, atol=0)

        def test_invalid_history_rejected_and_no_fake_simulated_depth_schedule(self):
            model = tiny_model()
            for tokens, depths, mode in [([7], [], 'early_exit'), ([7], [0], 'early_exit'),
                                         ([7], [5], 'early_exit'), ([7], [1], 'simulated')]:
                with self.assertRaises(ValueError):
                    list(replay_steps(model, torch.tensor([[4]]), tokens, depths,
                                      threshold=.5, execution_mode=mode))


if __name__ == '__main__':
    unittest.main()
