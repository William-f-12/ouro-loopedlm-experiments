"""Scientific invariants of quantization, depth accounting, and paired evaluation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import DynamicCache
from ouro_local.configuration_ouro import OuroConfig
from ouro_local.modeling_ouro import OuroForCausalLM
from experiments.quantization import nvfp4_qdq, quantize_core_mlp_, round_e2m1
from experiments.generation import generate_traced, select_depth
from experiments.scoring import grade_answer, grade_token_ids
from experiments.reporting import summarize, prepare_run


def tiny_model():
    torch.manual_seed(17)
    config = OuroConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=128, total_ut_steps=4,
        early_exit_threshold=1.0, layer_types=["full_attention"] * 2,
        bos_token_id=0, eos_token_id=0, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return OuroForCausalLM(config).eval()


class QuantizationTests(unittest.TestCase):
    def test_rounds_e2m1_ties_to_even_and_saturates(self):
        x = torch.tensor([0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 9])
        expected = torch.tensor([0, 0, 1, 1, 2, 2, 4, 4, 6.])
        torch.testing.assert_close(round_e2m1(x), expected)
        torch.testing.assert_close(round_e2m1(-x), -expected)

    def test_known_nvfp4_blocks_and_small_scales(self):
        # Tensor max=2688 -> global scale=1. First block scale=448 exactly.
        first = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2) * 448
        second = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2) * 2
        x = torch.stack([first, second, torch.zeros(16)])
        torch.testing.assert_close(nvfp4_qdq(x), x)
        tiny = torch.full((1, 16), 1e-9)
        mixed = torch.cat([x, tiny])
        self.assertTrue(torch.isfinite(nvfp4_qdq(mixed)).all())
        torch.testing.assert_close(nvfp4_qdq(torch.zeros_like(x)), torch.zeros_like(x))
        with self.assertRaises(ValueError):
            nvfp4_qdq(torch.zeros(2, 17))

    def test_only_core_mlp_weights_change(self):
        model = tiny_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        manifest = quantize_core_mlp_(model)
        changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
        expected = {f"model.layers.{i}.mlp.{p}.weight"
                    for i in range(2) for p in ("gate_proj", "up_proj", "down_proj")}
        self.assertEqual(changed, expected)
        self.assertEqual(len(manifest["modules"]), 6)
        self.assertFalse(manifest["packed_weights"])

    def test_fp8_block_scale_rounding_and_bf16_storage(self):
        x = torch.cat([torch.full((1, 16), 2688.), torch.full((1, 16), 61.8)])
        # global=1, scales=[448,10.3] -> E4M3 scales=[448,10].
        torch.testing.assert_close(nvfp4_qdq(x), torch.cat([torch.full((1, 16), 2688.), torch.full((1, 16), 60.)]))
        bf16 = torch.randn(16, 32).bfloat16()
        restored = nvfp4_qdq(bf16)
        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertTrue(restored.isfinite().all())

    def test_chunking_preserves_one_global_scale_per_matrix(self):
        small_chunks, large_chunks = tiny_model(), tiny_model()
        quantize_core_mlp_(small_chunks, chunk_rows=1)
        quantize_core_mlp_(large_chunks, chunk_rows=128)
        for name, tensor in small_chunks.state_dict().items():
            torch.testing.assert_close(tensor, large_chunks.state_dict()[name], rtol=0, atol=0)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_threshold_uses_conditional_hazards_and_final_remainder(self):
        # Exit PDF = [.25, .375, .1875, .1875].
        gates = [torch.tensor([[[v]]]) for v in (-1.0986123, 0., 0., 10.)]
        depth, pdf = select_depth(gates, .5)
        self.assertEqual(depth, 2)
        torch.testing.assert_close(pdf, torch.tensor([.25, .375, .1875, .1875]))
        self.assertEqual(select_depth(gates, 1.0)[0], 4)

    def test_trace_matches_upstream_tokens_and_counts_prefill_once(self):
        model = tiny_model()
        ids = torch.tensor([[4, 5, 6]])
        for threshold in (1.0, .5):
            model.early_exit_threshold = threshold
            options = {"exit_at_step": 3} if threshold == 1.0 else {}
            with torch.inference_mode():
                expected = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    past_key_values=DynamicCache(), max_new_tokens=5,
                    eos_token_id=None, pad_token_id=0, do_sample=False,
                    logits_to_keep=1, use_cache=True, **options)[0, ids.shape[1]:].tolist()
            result = generate_traced(model, ids, threshold=threshold, max_new_tokens=5, eos_ids=[])
            self.assertEqual(result["token_ids"], expected)
            self.assertEqual(len(result["selected_depths"]), 5)
            self.assertEqual(result["executed_depths"], [4] * 5)
            self.assertEqual(result["executed_core_steps"], 20)
            self.assertEqual(result["actual_forward_token_loops"], (3 + 4) * 4)
            self.assertTrue(result["truncated"])
            if threshold == 1.0:
                self.assertEqual(result["selected_depths"], [4] * 5)
        self.assertEqual(len(model.model._forward_hooks), 0)

    def test_eos_is_counted_once_even_at_token_limit(self):
        model = tiny_model()
        ids = torch.tensor([[4, 5]])
        first = generate_traced(model, ids, threshold=1.0, max_new_tokens=1, eos_ids=[])["token_ids"][0]
        result = generate_traced(model, ids, threshold=1.0, max_new_tokens=1, eos_ids=[first])
        self.assertEqual(result["token_ids"], [first])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["selected_core_steps"], 4)

    def test_bf16_saturated_gate_does_not_override_fixed_four_loop_reference(self):
        model = tiny_model().to(torch.bfloat16)
        with torch.no_grad():
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(100)
        ids = torch.tensor([[4, 5]])
        fixed = generate_traced(model, ids, threshold=1.0, max_new_tokens=2, eos_ids=[])
        adaptive = generate_traced(model, ids, threshold=.5, max_new_tokens=2, eos_ids=[])
        self.assertEqual(fixed["selected_depths"], [4, 4])
        self.assertEqual(adaptive["selected_depths"], [1, 1])
        self.assertEqual(fixed["executed_depths"], [4, 4])
        self.assertEqual(adaptive["executed_depths"], [4, 4])


class ScoringAndReportingTests(unittest.TestCase):
    def test_special_token_decoding_preserves_thinking_boundary_for_grading(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        vocab = {"<unk>": 0, "<think>": 1, "</think>": 2,
                 r"\boxed{42}": 3, "Cannot answer": 4, "<|im_end|>": 5}
        token = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=token, unk_token="<unk>",
                    eos_token="<|im_end|>", additional_special_tokens=["<think>", "</think>"])
        result = grade_token_ids("gsm8k", tokenizer, [1, 3, 2, 4, 5], "#### 42")
        self.assertIn("</think>", result["generated_text_raw"])
        self.assertNotIn("</think>", result["generated_text"])
        self.assertFalse(result["correct"])

    def test_thinking_boundary_and_unfinished_reasoning_are_not_final_answers(self):
        for text in [r"Reasoning: \boxed{42}",
                     r"Reasoning: \boxed{42}</think>I cannot solve it.",
                     r"Reasoning: \boxed{42}</think>Final answer: 41<|im_end|>"]:
            with self.subTest(text=text):
                self.assertFalse(grade_answer("gsm8k", text, "#### 42", require_thinking_end=True)["correct"])
        text = r"Reasoning: \boxed{41}</think>Final answer: 42<|im_end|>"
        self.assertTrue(grade_answer("gsm8k", text, "#### 42", require_thinking_end=True)["correct"])

    def test_math_symbolic_equivalence_and_wrong_answer(self):
        # Windows Math-Verify timeout uses an unpicklable nested worker;
        # only these offline fixtures disable timeouts. Colab keeps 5 seconds.
        self.assertTrue(grade_answer("math500", r"\boxed{\frac{1}{2}}", "0.5", timeout_seconds=0)["correct"])
        self.assertTrue(grade_answer("math500", r"\boxed{\sqrt{4}}", "2", timeout_seconds=0)["correct"])
        self.assertFalse(grade_answer("math500", r"\boxed{3}", "2", timeout_seconds=0)["correct"])

    def test_numeric_answer_extraction_is_exact_and_not_last_reasoning_number(self):
        cases = [
            (r"Reasoning 50. </think> Final answer: \boxed{1,200}", "work\n#### 1200", True),
            (r"\boxed{\frac{1}{2}}", "#### 0.5", True),
            (r"\boxed{1\,200}", "#### 1200", True),
            (r"\boxed{1,2}", "#### 12", False),
            ("Final answer: -3", "#### -3", True),
            ("We still need to add 42", "#### 42", False),
            (r"\boxed{1201}", "#### 1200", False),
            (r"\boxed{0.500000001}", "#### 0.5", False),
        ]
        for text, gold, correct in cases:
            with self.subTest(text=text):
                self.assertEqual(grade_answer("gsm8k", text, gold)["correct"], correct)

    def test_pairing_uses_only_jointly_solved_questions(self):
        records = []
        for arm, corrects, counts in [("bf16", [True, False], [8, 4]),
                                     ("ptq_nvfp4", [True, True], [12, 100])]:
            for i, (correct, count) in enumerate(zip(corrects, counts)):
                records.append(dict(arm=arm, dataset="gsm8k", example_id=str(i),
                    threshold=.5, correct=correct, generated_tokens=4, selected_core_steps=count,
                    executed_core_steps=16, actual_forward_token_loops=24,
                    selected_depths=[count / 4] * 4, executed_depths=[4] * 4,
                    truncated=False, parse_success=True, generation_seconds=1))
        report = summarize(records)
        pair = report["paired"][0]
        self.assertEqual(pair["jointly_solved"], 1)
        self.assertEqual(pair["selected_compute_inflation_ratio_of_sums"], .5)
        self.assertIsNone(pair["selected_compute_inflation_per_problem"]["se"])

    def test_resume_rejects_changed_protocol_and_nonresume_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            prepare_run(path, {"seed": 7}, resume=False)
            with self.assertRaises(FileExistsError):
                prepare_run(path, {"seed": 7}, resume=False)
            prepare_run(path, {"seed": 7}, resume=True)
            with self.assertRaisesRegex(ValueError, "protocol|manifest|different"):
                prepare_run(path, {"seed": 8}, resume=True)


if __name__ == "__main__":
    unittest.main()
