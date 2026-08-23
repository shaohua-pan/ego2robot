#!/usr/bin/env bash
# Create the three conda environments the pipeline needs.
#
# Three environments are unavoidable: WiLoR + SAM 3 need torch 2.6 and
# transformers >= 4.58, Dyn-HaMR is pinned to torch 1.13 / numpy 1.23, and VIPE's
# vendored GroundingDINO breaks on transformers 5.x. Nothing here is shared.
#
# Set PIP_INDEX_URL / http_proxy beforehand if your network needs them.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
DYNHAMR=${EGO2ROBOT_STORE:-/root/paddlejob/ego}/third_party/Dyn-HaMR
step() { echo; echo "############ $* ############"; }

step "1/3 ego2robot: WiLoR (stage 1) + SAM 3 + stages 3-5"
conda create -n ego2robot python=3.11 -y
EP=$(conda info --base)/envs/ego2robot/bin/python
$EP -m pip install "torch==2.6.0" "torchvision==0.21.0"
# WiLoR's requirements minus the demo-only extras (gradio, webdataset).
$EP -m pip install numpy opencv-python pytorch-lightning scikit-image \
    "smplx==0.1.28" yacs timm einops pandas hydra-core rich "ultralytics==8.1.34"
$EP -m pip install --no-build-isolation git+https://github.com/mattloper/chumpy
# av decodes AV1 (the opencv wheel cannot), h5py stores the tracks,
# transformers >= 4.58 ships Sam3VideoModel, accelerate does its device placement.
$EP -m pip install av h5py "transformers>=4.58" accelerate
# ProPainter (stage 4) and MuJoCo + mink with the quadprog backend A.4 specifies (stage 5).
$EP -m pip install addict yapf imageio-ffmpeg
$EP -m pip install "mujoco==3.11.0" "mink==1.3.0" "qpsolvers[quadprog]" pyyaml
# Stage 6 reads VIPE's half-float EXR depth and renders through Mesa when there is no EGL:
#   apt-get install libosmesa6   (then run with MUJOCO_GL=osmesa)
$EP -m pip install OpenEXR
$EP -c "
import torch, transformers, chumpy, av, h5py, ultralytics, mujoco, mink, qpsolvers
print('torch', torch.__version__, '| transformers', transformers.__version__)
print('SAM 3 available:', hasattr(transformers, 'Sam3VideoModel'))
print('mujoco', mujoco.__version__, '| quadprog:', 'quadprog' in qpsolvers.available_solvers)"

step "2/3 dynhamr: temporal optimization (paper eq. 6)"
conda create -n dynhamr python=3.10 -y
DP=$(conda info --base)/envs/dynhamr/bin/python
$DP -m pip install "torch==1.13.0+cu117" "torchvision==0.14.0+cu117" \
    --extra-index-url https://download.pytorch.org/whl/cu117
# torch 1.13 needs numpy < 2 and chumpy needs the legacy setuptools; both get
# silently upgraded by later installs, so keep the constraints file in front of
# every pip call in this environment.
$DP -m pip install "numpy==1.23.5" "setuptools==59.5.0"
$DP -m pip install -c scripts/constraints-dynhamr.txt scipy opencv-python matplotlib \
    imageio imageio-ffmpeg joblib tqdm pyyaml hydra-core omegaconf smplx trimesh \
    dill einops loguru pandas h5py plyfile tensorboard "torchgeometry==0.1.2"
$DP -m pip install -c scripts/constraints-dynhamr.txt --no-build-isolation \
    git+https://github.com/nghorbani/configer human-body-prior \
    git+https://github.com/mattloper/chumpy git+https://github.com/otaheri/MANO
$DP -c "
import torch, numpy, chumpy, smplx, mano, h5py
print('torch', torch.__version__, '| numpy', numpy.__version__, '| deps OK')"

step "3/3 vipe: camera intrinsics + trajectory"
conda create -n vipe python=3.10 -y
VP=$(conda info --base)/envs/vipe/bin/python
# vipe's csrc includes <eigen3/Eigen/Dense>, and pip builds outside the env.
conda install -n vipe -y -c conda-forge eigen
export CPATH=$(conda info --base)/envs/vipe/include:${CPATH:-}
$VP -m pip install "torch==2.6.0" "torchvision==0.21.0"
# hydra-core is missing from vipe's own dependency list; transformers must stay
# on 4.x or its vendored GroundingDINO fails with a BertModel attribute error.
$VP -m pip install hydra-core "transformers==4.51.3"
$VP -m pip install --no-build-isolation -e "$DYNHAMR/third-party/vipe"
$VP -c "import vipe, vipe_ext; print('vipe + vipe_ext OK')"
unset CPATH

step "DONE"
