"""Minimal LoRA for the SANA-WM stage-1 DiT.

Hand-rolled rather than peft-wrapped: the WM DiT has a custom forward signature
and runs under context parallel, and module renaming breaks both. Adapters are
saved in peft's on-disk layout so any peft-compatible loader can merge them.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_linear", "kv_linear", "proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        h = self.dropout(x)
        delta = (h.to(self.lora_A.dtype) @ self.lora_A.T) @ self.lora_B.T
        return self.base(x) + delta.to(x.dtype) * self.scale


def _matches(name: str, targets, pattern: str | None) -> bool:
    if pattern:
        return re.search(pattern, name) is not None
    return name.rsplit(".", 1)[-1] in targets


def inject(model: nn.Module, r: int = 32, alpha: float = 64.0, dropout: float = 0.0,
           targets=DEFAULT_TARGETS, pattern: str | None = None) -> list[str]:
    """Freeze the base model and wrap every matching nn.Linear with a LoRA branch."""
    for p in model.parameters():
        p.requires_grad_(False)
    hits = [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and _matches(n, targets, pattern)]
    if not hits:
        raise RuntimeError(f"LoRA matched no modules (targets={targets}, pattern={pattern})")
    for name, mod in hits:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1],
                LoRALinear(mod, r, alpha, dropout).to(mod.weight.device, mod.weight.dtype))
    return [n for n, _ in hits]


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _clean(name: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        while name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("._orig_mod.", ".").replace(".module.", ".")


def save_adapter(model: nn.Module, out_dir: str | Path, r: int, alpha: float,
                 targets=DEFAULT_TARGETS, pattern: str | None = None) -> Path:
    """Write adapter_config.json + adapter_model.safetensors in peft layout."""
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            key = _clean(name)
            sd[f"base_model.model.{key}.lora_A.weight"] = mod.lora_A.detach().float().cpu()
            sd[f"base_model.model.{key}.lora_B.weight"] = mod.lora_B.detach().float().cpu()
    if not sd:
        raise RuntimeError("save_adapter found no LoRALinear modules")
    save_file(sd, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": r, "lora_alpha": alpha,
        "target_modules": list(targets), "target_pattern": pattern,
        "n_modules": len(sd) // 2,
    }, indent=2))
    return out
