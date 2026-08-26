"""Minimal LoRA for the SANA-WM stage-1 DiT.

Implemented as parameters registered on the existing ``nn.Linear`` plus a forward
hook, NOT as a wrapper module. Replacing the module changes the tree that FSDP2's
auto-wrap and the context-parallel machinery introspect, which makes gradient
checkpointing recompute a differently-shaped parameter and raises CheckpointError.
Keeping module identity avoids that entirely.

Adapters are written in peft's on-disk layout so any peft-compatible loader can
merge them.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_linear", "kv_linear", "proj")


def _lora_forward_hook(module: nn.Linear, args, output):
    x = args[0]
    h = module.lora_dropout(x) if module.lora_dropout is not None else x
    delta = (h @ module.lora_A.transpose(0, 1)) @ module.lora_B.transpose(0, 1)
    return output + delta * module.lora_scale


def _matches(name: str, targets, pattern: str | None) -> bool:
    if pattern:
        return re.search(pattern, name) is not None
    return name.rsplit(".", 1)[-1] in targets


def is_lora(module: nn.Module) -> bool:
    return hasattr(module, "lora_A") and hasattr(module, "lora_B")


def inject(model: nn.Module, r: int = 32, alpha: float = 64.0, dropout: float = 0.0,
           targets=DEFAULT_TARGETS, pattern: str | None = None) -> list[str]:
    """Freeze the base model and attach a LoRA branch to every matching nn.Linear."""
    for p in model.parameters():
        p.requires_grad_(False)
    hits = [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not is_lora(m) and _matches(n, targets, pattern)]
    if not hits:
        raise RuntimeError(f"LoRA matched no modules (targets={targets}, pattern={pattern})")
    for _, mod in hits:
        dev, dtype = mod.weight.device, mod.weight.dtype
        a = nn.Parameter(torch.zeros(r, mod.in_features, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        mod.register_parameter("lora_A", a)
        mod.register_parameter("lora_B", nn.Parameter(
            torch.zeros(mod.out_features, r, device=dev, dtype=dtype)))
        mod.lora_scale = alpha / r
        mod.lora_dropout = nn.Dropout(dropout) if dropout > 0 else None
        mod.register_forward_hook(_lora_forward_hook)
    return [n for n, _ in hits]


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _clean(name: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        while name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("._orig_mod.", ".").replace(".module.", ".")


def _gather(t: torch.Tensor) -> torch.Tensor:
    """FSDP2 shards parameters as DTensors; full_tensor() is a collective, so every
    rank must reach it even though only rank 0 writes the file."""
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    return t.detach().float().cpu()


def save_adapter(model: nn.Module, out_dir: str | Path, r: int, alpha: float,
                 targets=DEFAULT_TARGETS, pattern: str | None = None) -> Path:
    """Write adapter_config.json + adapter_model.safetensors in peft layout."""
    import torch.distributed as dist
    from safetensors.torch import save_file

    out = Path(out_dir)
    sd = {}
    for name, mod in model.named_modules():
        if is_lora(mod):
            key = _clean(name)
            sd[f"base_model.model.{key}.lora_A.weight"] = _gather(mod.lora_A)
            sd[f"base_model.model.{key}.lora_B.weight"] = _gather(mod.lora_B)
    if not sd:
        raise RuntimeError("save_adapter found no LoRA-augmented modules")
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return out
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": r, "lora_alpha": alpha,
        "target_modules": list(targets), "target_pattern": pattern,
        "n_modules": len(sd) // 2,
    }, indent=2))
    return out
