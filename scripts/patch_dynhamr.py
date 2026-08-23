"""Patch the Dyn-HaMR checkout to the values the Ego2Robot paper specifies.

Two upstream defaults do not match paper appendix A.1 / eq.(6):

1. ``dyn-hamr/optim/losses.py`` calls ``depth_constraint_loss`` with
   ``min_depth=0.0, max_depth=999``, i.e. it only keeps the hand in front of the
   camera. The paper constrains hand depth to [0.05, 0.4] m.
2. ``dyn-hamr/confs/optim.yaml`` disables the biomechanical loss
   (``bio: [0.0, 0.0, 0.0]``), while eq.(6) is
   ``L_dyn = L_data + lambda_smooth L_smooth + lambda_bio L_bio``. The weight is
   set to the value the Dyn-HaMR author left commented out on that same line
   (``[10, 10, 10]``); the paper does not give a number.

The patch is idempotent, so it is safe to re-run after re-extracting Dyn-HaMR.

Usage:
    python scripts/patch_dynhamr.py [--root /root/paddlejob/ego/third_party/Dyn-HaMR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config  # noqa: E402

DEPTH_OLD = """            cur_loss = depth_constraint_loss(
                pred_data["joints3d"],
                cam_R,
                cam_t,
                min_depth=0.0,
                max_depth=999
            )"""

DEPTH_NEW = """            cur_loss = depth_constraint_loss(
                pred_data["joints3d"],
                cam_R,
                cam_t,
                # Ego2Robot paper A.1: hand depth is constrained to [0.05, 0.4] m.
                min_depth={dmin},
                max_depth={dmax},
            )"""

BIO_OLD = "    bio : [0.0, 0.0, 0.0] # [10, 10, 10]"
BIO_NEW = "    bio : [10, 10, 10] # Ego2Robot eq.(6) enables L_bio (upstream reference value)"


def patch_file(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text()
    if new in text:
        print(f"[patch] {label}: already applied")
        return False
    if old not in text:
        raise SystemExit(f"[patch] {label}: expected snippet not found in {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"[patch] {label}: patched {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=config.DYNHAMR_ROOT)
    args = parser.parse_args()

    losses = args.root / "dyn-hamr" / "optim" / "losses.py"
    optim_conf = args.root / "dyn-hamr" / "confs" / "optim.yaml"
    for f in (losses, optim_conf):
        if not f.is_file():
            raise SystemExit(f"[patch] not found: {f}")

    patch_file(losses, DEPTH_OLD,
               DEPTH_NEW.format(dmin=config.HAND_DEPTH_MIN_M, dmax=config.HAND_DEPTH_MAX_M),
               "depth constraint [0.05, 0.4] m")
    patch_file(optim_conf, BIO_OLD, BIO_NEW, "L_bio weight")


if __name__ == "__main__":
    main()
