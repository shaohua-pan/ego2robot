"""Run Dyn-HaMR's temporal optimization headlessly.

``run_opt.py`` imports ``vis/viewer.py`` at module load, which imports pyrender,
which initialises PyOpenGL against EGL. This machine has no EGL/OSMesa, so the
import fails before any optimization starts. The paper only needs the optimized
hand parameters (eq.(6)); rendering happens later in MuJoCo, so pyrender is
replaced with mock modules, the same trick ``pipeline/s1_hand_pose/wilor_runner.py``
uses for WiLoR.

Set EGO2ROBOT_ENABLE_PYRENDER=1 to keep the real package on hosts with a working
GL stack.

Usage (from anywhere):
    conda run -n dynhamr python scripts/run_dynhamr.py \
        data=ego2robot data.seq=egodex_sample is_static=False run_vis=False
"""
from __future__ import annotations

import os
import runpy
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from pathlib import Path
from unittest.mock import MagicMock

DYNHAMR_DIR = Path("/root/paddlejob/ego/third_party/Dyn-HaMR/dyn-hamr")
STUBBED_ROOTS = ("pyrender",)


class _MockLoader(Loader):
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module._ego2robot_stub = True
        module.__getattr__ = lambda name: MagicMock()  # PEP 562
        module.__path__ = []  # allow submodule imports
        return module

    def exec_module(self, module):
        pass


class _MockFinder(MetaPathFinder):
    """Resolve every ``pyrender`` / ``pyrender.*`` import to a mock module."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root not in STUBBED_ROOTS:
            return None
        from importlib.machinery import ModuleSpec

        return ModuleSpec(fullname, _MockLoader(), is_package=True)


def _stub_vposer() -> None:
    """Satisfy run_opt.py's VPoser imports without the v2 human_body_prior package.

    ``run_opt.py`` imports ``load_model`` and ``VPoser`` at module level, but the
    call that would use them is commented out upstream (``pose_prior = None``) and
    the paper's eq.(6) has no pose-prior term. The v2 package that provides those
    names requires Python >= 3.11 while Dyn-HaMR pins 3.10, so the two symbols are
    filled in with placeholders that raise if anything ever calls them.
    """
    import human_body_prior.tools.model_loader as model_loader

    def _unavailable(*args, **kwargs):
        raise RuntimeError(
            "VPoser is not installed: Ego2Robot's DynHaMR stage does not use a "
            "pose prior (paper eq.(6) = L_data + L_smooth + L_bio)")

    if not hasattr(model_loader, "load_model"):
        model_loader.load_model = _unavailable

    name = "human_body_prior.models.vposer_model"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.VPoser = _unavailable
        sys.modules[name] = module


def main() -> None:
    if os.environ.get("EGO2ROBOT_ENABLE_PYRENDER") != "1":
        sys.meta_path.insert(0, _MockFinder())
        print("[run_dynhamr] pyrender stubbed out (no EGL on this host)")

    os.chdir(DYNHAMR_DIR)
    sys.path.insert(0, str(DYNHAMR_DIR))
    sys.path.insert(0, str(DYNHAMR_DIR / "HMP"))
    _stub_vposer()
    # hydra reads sys.argv; keep the overrides, drop our script path.
    sys.argv = ["run_opt.py", *sys.argv[1:]]
    runpy.run_path(str(DYNHAMR_DIR / "run_opt.py"), run_name="__main__")


if __name__ == "__main__":
    main()
