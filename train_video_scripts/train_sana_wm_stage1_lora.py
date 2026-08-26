"""SANA-WM stage-1 LoRA finetuning.

Thin wrapper over train_sana_wm_stage1.py: same data path, optimizer and CP
settings, but the base DiT is frozen and only LoRA adapters train, so a node
costs tens of MB instead of tens of GB. Knobs live under `train.extra.lora`:

    train:
      extra:
        lora: {r: 32, alpha: 64, dropout: 0.0, targets: [q_linear, kv_linear, proj]}

Usage:
    torchrun --nproc_per_node=8 train_video_scripts/train_sana_wm_stage1_lora.py \
      --config_path configs/sana_wm/stage1/sana_wm_stage1_lora_base.yaml
"""
from __future__ import annotations

import os
import os.path as osp
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import importlib.util  # noqa: E402


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "train_sana_wm_stage1", HERE / "train_sana_wm_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _lora_cfg_from_argv() -> dict:
    """Read train.extra.lora straight from the YAML, before pyrallis parses it."""
    path = None
    for i, a in enumerate(sys.argv):
        if a == "--config_path" and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
        elif a.startswith("--config_path="):
            path = a.split("=", 1)[1]
    if not path:
        raise SystemExit("--config_path is required")
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    extra = ((cfg.get("train") or {}).get("extra") or {})
    lora = dict(extra.get("lora") or {})
    lora.setdefault("r", 32)
    lora.setdefault("alpha", 64.0)
    lora.setdefault("dropout", 0.0)
    lora.setdefault("targets", ["q_linear", "kv_linear", "proj"])
    lora.setdefault("pattern", None)
    return lora


def main() -> None:
    from diffusion.utils.lora_wm import inject, save_adapter

    base = _load_base()
    lora = _lora_cfg_from_argv()
    state = {"injected": False, "names": []}

    orig_build_optimizer = base.build_optimizer

    def build_optimizer(model, optimizer_cfg):
        if not state["injected"]:
            state["names"] = inject(model, r=int(lora["r"]), alpha=float(lora["alpha"]),
                                    dropout=float(lora["dropout"]),
                                    targets=tuple(lora["targets"]), pattern=lora["pattern"])
            state["injected"] = True
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in model.parameters())
            print(f"[lora] wrapped {len(state['names'])} modules; "
                  f"trainable {n_train/1e6:.2f}M / {n_all/1e6:.1f}M params", flush=True)
        return orig_build_optimizer(model, optimizer_cfg)

    def save_checkpoint(work_dir, epoch, model, accelerator=None, step=None, **kw):
        model = accelerator.unwrap_model(model) if accelerator is not None else model
        import torch.distributed as dist

        out = Path(work_dir).parent / "lora" / f"step_{step or 0}"
        save_adapter(model, out, r=int(lora["r"]), alpha=float(lora["alpha"]),
                     targets=tuple(lora["targets"]), pattern=lora["pattern"])
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return str(out)
        latest = Path(work_dir).parent / "lora" / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.name)
        print(f"[lora] saved adapter -> {out}", flush=True)
        return str(out)

    base.build_optimizer = build_optimizer
    base.save_checkpoint = save_checkpoint
    base.main()


if __name__ == "__main__":
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    main()
