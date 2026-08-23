"""Generate the Dyn-HaMR BMC assets with Hand-BMC-pytorch under a headless env.

Dyn-HaMR's biomechanical loss (paper eq.(6) L_bio) loads seven .npy files from
`_DATA/BMC/`, which upstream expects the user to compute with
https://github.com/MengHao666/Hand-BMC-pytorch. Two things break when running
`calculate_convex_hull.py` on a modern headless machine:

1. it maximizes the matplotlib window, which needs an interactive backend;
2. it calls np.array() on a ragged list of hulls, which NumPy >= 1.24 rejects
   unless dtype=object is passed.

This driver stubs around both without modifying the upstream checkout.

Usage:
    python scripts/prepare_bmc.py --repo /path/to/Hand-BMC-pytorch
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class _StubWindow:
    def showMaximized(self):
        pass


class _StubManager:
    window = _StubWindow()


def _patch_matplotlib() -> None:
    plt.get_current_fig_manager = lambda *args, **kwargs: _StubManager()


def _patch_numpy_array() -> None:
    """Retry np.array(..., dtype=object) when the input is ragged."""
    original = np.array

    def patched(obj, *args, **kwargs):
        try:
            return original(obj, *args, **kwargs)
        except ValueError as err:
            if "inhomogeneous" not in str(err) or "dtype" in kwargs:
                raise
            return original(obj, *args, dtype=object, **kwargs)

    np.array = patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Hand-BMC-pytorch checkout")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    os.chdir(repo)
    # The upstream scripts import their sibling modules (config, utils) directly.
    sys.path.insert(0, str(repo))

    _patch_matplotlib()
    _patch_numpy_array()

    for script in ("calculate_bmc.py", "calculate_convex_hull.py"):
        path = repo / script
        print(f"[bmc] running {script}")
        # The upstream scripts parse sys.argv themselves; hide our own flags.
        sys.argv = [script]
        code = compile(path.read_text(), script, "exec")
        exec(code, {"__name__": "__main__", "__file__": str(path)})

    produced = sorted(p.name for p in (repo / "BMC").glob("*.npy"))
    print(f"[bmc] BMC/ contains {len(produced)} files: {produced}")


if __name__ == "__main__":
    main()
