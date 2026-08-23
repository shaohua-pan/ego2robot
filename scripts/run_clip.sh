#!/usr/bin/env bash
# Run stages 1-6 on one clip, in the order the README documents.
#
#   [CUDA_VISIBLE_DEVICES=n] bash scripts/run_clip.sh <video.mp4> [robot ...]
#
# Every stage writes under $STORE/outputs/s<N>_<clip name>/ and every third-party
# artifact is keyed by the clip name, so several clips can run at once - one per GPU,
# with CUDA_VISIBLE_DEVICES picking the card. Stage 5 and 6 run once per robot.
#
# The clip must be a single continuous shot: VIPE's camera track, Dyn-HaMR's temporal
# optimization and the single base pose of eq.(4) are all only meaningful within one
# take, and a cut also costs time - the 120-frame window that straddles the sample
# clip's frame-94 cut took VIPE 83 minutes against 43 for a longer cut-free clip.
set -euo pipefail

# Every checkpoint is already in the local cache; without this, huggingface_hub spends
# minutes retrying HEAD requests before falling back to it.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

VIDEO=${1:?usage: run_clip.sh <video.mp4> [robot ...]}
shift || true
ROBOTS=("${@:-panda}")

NAME=$(basename "$VIDEO" .mp4)
# conda is not always on PATH in a non-interactive shell; every stage is invoked
# through its env's interpreter directly, so only the envs directory is needed.
CONDA=${EGO2ROBOT_CONDA_ENVS:-$( (conda info --base 2>/dev/null) || echo "$HOME/anaconda3" )/envs}
STORE=${EGO2ROBOT_STORE:-/root/paddlejob/ego}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DYNHAMR=$STORE/third_party/Dyn-HaMR
VIPE_OUT=$DYNHAMR/third-party/vipe/vipe_results

OUT1=$STORE/outputs/s1_$NAME
OUT3=$STORE/outputs/s3_$NAME
OUT4=$STORE/outputs/s4_$NAME
OUT5=$STORE/outputs/s5_$NAME
H5=$OUT1/hand_pose.h5

cd "$REPO"

echo "########## stage 1: WiLoR + association ($NAME)"
"$CONDA"/ego2robot/bin/python -m pipeline.s1_hand_pose.run_stage1 \
    --video "$VIDEO" --out-dir "$OUT1"

echo "########## stage 1: export for Dyn-HaMR"
"$CONDA"/ego2robot/bin/python -m pipeline.s1_hand_pose.dynhamr_bridge \
    --h5 "$H5" --root "$STORE"/outputs/dynhamr_data

echo "########## stage 1: VIPE (intrinsics, camera trajectory, metric depth)"
( cd "$DYNHAMR"/third-party/vipe && "$CONDA"/vipe/bin/vipe infer \
    "$STORE"/outputs/dynhamr_data/videos/"$NAME".mp4 )

echo "########## stage 1: re-export with VIPE's focal length, then eq.(6)"
"$CONDA"/ego2robot/bin/python -m pipeline.s1_hand_pose.dynhamr_bridge \
    --h5 "$H5" --root "$STORE"/outputs/dynhamr_data \
    --vipe-intrinsics "$VIPE_OUT"/intrinsics/"$NAME".npz
"$CONDA"/dynhamr/bin/python scripts/run_dynhamr.py \
    data=ego2robot "data.seq=$NAME" is_static=False run_vis=False

RUN_DIR=$(ls -d "$DYNHAMR"/outputs/logs/video-custom/*/"$NAME"-all-shot-0-0--1 | tail -1)
echo "########## stage 1: import $RUN_DIR"
"$CONDA"/dynhamr/bin/python -m pipeline.s1_hand_pose.dynhamr_import \
    --run-dir "$RUN_DIR" --h5 "$H5"

echo "########## stage 2: retarget + smooth"
"$CONDA"/dynhamr/bin/python -m pipeline.s2_retarget.run_stage2 --h5 "$H5"

echo "########## stage 3: arm segmentation"
"$CONDA"/ego2robot/bin/python -m pipeline.s3_arm_seg.run_stage3 \
    --video "$VIDEO" --out-dir "$OUT3"

echo "########## stage 4: hand removal"
"$CONDA"/ego2robot/bin/python -m pipeline.s4_hand_removal.run_stage4 \
    --video "$VIDEO" --mask-dir "$OUT3"/arm_mask --out-dir "$OUT4" --resize-ratio 0.5

for ROBOT in "${ROBOTS[@]}"; do
    echo "########## stage 5: base search + IK ($ROBOT)"
    "$CONDA"/ego2robot/bin/python -m pipeline.s5_base_ik.run_stage5 \
        --h5 "$H5" --robot "$ROBOT" --out-dir "$OUT5"

    echo "########## stage 6: depth compositing ($ROBOT)"
    MUJOCO_GL=osmesa "$CONDA"/ego2robot/bin/python -m pipeline.s6_composite.run_stage6 \
        --stage5 "$OUT5"/robot_"$ROBOT".h5 --inpainted "$OUT4"/inpainted \
        --mask-dir "$OUT3"/arm_mask --depth "$VIPE_OUT"/depth/"$NAME".zip \
        --out-dir "$STORE"/outputs/s6_"$NAME"_"$ROBOT"
done

echo "########## done: $NAME"
