#!/usr/bin/env bash
# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.

# Zero-shot benchmarks of PINE-B at 512x512, one sampling step, EMA weights, with the protocols of the paper:
#   point map and depth  DIODE, KITTI, NYUv2, ETH3D, HAMMER, iBims-1, Booster (3,444 images), with and without intrinsics
#   normals              NYUv2, ScanNet, iBims-1, Sintel (full-frame DSINE protocol), with and without intrinsics
#   instances            COCOexc, DRAM, EgoHOS, iShape, PIDRay
#   field of view        the seven point map sets, without intrinsics
# Results go to OUTPUT_DIR/{point,normal,fov}/ and OUTPUT_DIR/instance.json.
# scripts/eval_ibims_planarity.py and scripts/eval_inference_efficiency.py give the two other tables of the paper.

cd "$(dirname "$0")/.."

CHECKPOINT=${CHECKPOINT:-pretrained/pine-b-512.pth}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/pine-b-512/eval}
EVAL_DATA=datasets/eval
# The frozen DINOv3 encoder is not part of the checkpoint: its weights are gated, so
# they are downloaded separately into pretrained/dinov3/ (see MODELS.md).
export DINOV3_WEIGHTS_DIR=${DINOV3_WEIGHTS_DIR:-pretrained/dinov3}
export DSINE_DATA_ROOT=${DSINE_DATA_ROOT:-${EVAL_DATA}/normal}
export DSINE_SPLIT_ROOT=${DSINE_SPLIT_ROOT:-${EVAL_DATA}/normal_splits}
export OPENCV_IO_ENABLE_OPENEXR=1

python scripts/eval_point.py --checkpoint ${CHECKPOINT} --data-root ${EVAL_DATA} --output-dir ${OUTPUT_DIR}/point
python scripts/eval_normal.py --checkpoint ${CHECKPOINT} --output-dir ${OUTPUT_DIR}/normal
python scripts/eval_instance.py --checkpoint ${CHECKPOINT} --eval-root ${EVAL_DATA}/instance_segmentation \
    --output-dir ${OUTPUT_DIR}
python scripts/eval_camera_fov.py --checkpoint ${CHECKPOINT} --data-root ${EVAL_DATA} --output-dir ${OUTPUT_DIR}/fov
