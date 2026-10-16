"""Check the actual local model on CPU using tiny random weights; no downloads."""
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import transformers
    from transformers import DynamicCache
    from inference import build_config, source_provenance
    from ouro_local.configuration_ouro import OuroConfig
    from ouro_local.modeling_ouro import OuroForCausalLM

    assert transformers.__version__ == "4.57.1", "Install requirements.txt first"
    assert Path(inspect.getfile(OuroForCausalLM)).resolve() == ROOT / "ouro_local/modeling_ouro.py"
    baseline = build_config()
    assert baseline.num_hidden_layers == 24, "This smoke check targets the shipped 1.4B architecture"
    assert build_config(loops=3).total_ut_steps == 3
    torch.manual_seed(42)
    torch.set_num_threads(2)
    measurements = []
    for loops in (3, 4):
        config = OuroConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128, total_ut_steps=loops,
            layer_types=["full_attention"] * 2, bos_token_id=0, eos_token_id=0, pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        model = OuroForCausalLM(config).eval()
        ids = torch.tensor([[4, 5, 6, 7]])
        with torch.inference_mode():
            full = model(input_ids=ids, use_cache=False).logits
            cache = DynamicCache()
            pieces = [model(input_ids=ids[:, i:i+1], past_key_values=cache, use_cache=True).logits
                      for i in range(ids.shape[1])]
            cached = torch.cat(pieces, dim=1)
            torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-4)
            assert len(cache.layers) == loops * config.num_hidden_layers
            assert all(layer.keys.shape[2] == ids.shape[1] for layer in cache.layers)
            output = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=DynamicCache(),
                max_new_tokens=4, do_sample=False, use_cache=True, pad_token_id=0, logits_to_keep=1,
            )
            assert ids.shape[1] < output.shape[1] <= ids.shape[1] + 4
        measurements.append({"loops": loops, "cache_layers": len(cache.layers),
                             "max_logit_difference": (full - cached).abs().max().item()})
    print(json.dumps({"status": "PASS", "scope": "tiny random-weight CPU model; no pretrained GPU test",
                      "torch": torch.__version__, "transformers": transformers.__version__,
                      "measurements": measurements, "source": source_provenance()}, indent=2))


if __name__ == "__main__":
    main()
