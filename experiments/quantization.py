"""MLP-only NVFP4 RTN numerical simulation; no packed storage or low-bit GEMM.

Format: E2M1 weights, 16 contiguous input features per block, E4M3FN
block scales and a FP32 per-weight-matrix scale. Round to nearest, ties to even.
Reference: NVIDIA/Model-Optimizer modelopt/torch/quantization/qtensor/nvfp4_tensor.py.
"""
import torch


def round_e2m1(values):
    bounds = values.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    levels = values.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    magnitude = values.abs().contiguous()
    indices = torch.bucketize(magnitude, bounds)
    # At these midpoints the upper representable code has even mantissa.
    indices += ((magnitude == .75) | (magnitude == 1.75) | (magnitude == 3.5)).long()
    return levels[indices] * values.sign()


@torch.no_grad()
def nvfp4_qdq(weight, block_size=16, global_scale=None):
    if block_size != 16 or weight.ndim != 2 or weight.shape[-1] % block_size:
        raise ValueError("NVFP4 requires 2D weights with input features divisible by 16")
    work = weight.float()
    if not torch.isfinite(work).all():
        raise ValueError("Non-finite input weights")
    if global_scale is None:
        amax = work.abs().amax()
        global_scale = torch.where(amax > 0, amax / (6 * 448), torch.ones_like(amax))
    blocks = work.reshape(work.shape[0], -1, block_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / (6 * global_scale)
    # Positive minimum avoids underflow-to-zero; maximum avoids FP8 NaN.
    scale = scale.clamp(min=2**-9, max=448).to(torch.float8_e4m3fn).float()
    effective_scale = scale * global_scale
    restored = round_e2m1(blocks / effective_scale) * effective_scale
    return restored.reshape_as(weight).to(weight.dtype)


@torch.no_grad()
def quantize_core_mlp_(model, chunk_rows=128):
    """Materialize RTN-dequantized weights in place once; keep activation dtype."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    manifest = {
        "format": "NVFP4_E2M1_E4M3FN_RTN", "mode": "weight_only_qdq",
        "block_size": 16, "block_axis": -1, "global_scale": "FP32 per matrix",
        "rounding": "nearest_ties_to_even", "packed_weights": False,
        "scope": "shared core MLP gate_proj/up_proj/down_proj only", "modules": [],
    }
    for layer_index, layer in enumerate(model.model.layers):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp, projection)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"Expected Linear at core MLP {layer_index}/{projection}")
            weight = module.weight
            if weight.shape[-1] % 16:
                raise ValueError("MLP input dimension must be divisible by 16")
            amax = weight.detach().abs().amax().float()
            scale = torch.where(amax > 0, amax / (6 * 448), torch.ones_like(amax))
            squared_error = torch.zeros((), device=weight.device, dtype=torch.float64)
            squared_weight = torch.zeros_like(squared_error)
            for start in range(0, weight.shape[0], chunk_rows):
                part = weight[start:start + chunk_rows]
                restored = nvfp4_qdq(part, global_scale=scale)
                squared_error += (part.float() - restored.float()).square().sum().double()
                squared_weight += part.float().square().sum().double()
                part.copy_(restored)
            manifest["modules"].append({
                "name": f"model.layers.{layer_index}.mlp.{projection}",
                "shape": list(weight.shape), "storage_dtype": str(weight.dtype),
                "global_scale": scale.item(), "mse": squared_error.item() / weight.numel(),
                "relative_squared_error": (squared_error / squared_weight).item()
                    if squared_weight.item() > 0 else 0.,
            })
    if not manifest["modules"]:
        raise ValueError("No core MLP modules found")
    return manifest
