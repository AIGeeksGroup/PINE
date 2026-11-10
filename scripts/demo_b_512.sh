#!/usr/bin/env bash
# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.

# PINE-B (512x512) on your own images. Set IMAGE_DIR below; .jpg/.png, nested dirs are fine.
# Results stay under OUTPUT_DIR/eval/<checkpoint>/<weights>/point/<image folder>/.
# Each output is named after its input: sample0.png -> sample0_depth.png, sample0_normal.png,
# sample0_instance.png and sample0_pointcloud.ply.

# Any input resolution works: the full frame is resized (aspect ratio kept) to the
# resolution closest to the 32x32 tokens the model was trained on, the prediction runs
# there, then it is resized back to the input resolution for saving. Retarget the token
# budget with --eval_wild_target_tokens.
# The images have no intrinsics, so the model runs with its unknown-camera condition.

cd "$(dirname "$0")/.."

IMAGE_DIR=${IMAGE_DIR:-assets/demo}
CHECKPOINT=${CHECKPOINT:-checkpoints/pine-b-512/checkpoint-last.pth}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/pine-b-512}
# The frozen DINOv3 encoder is not part of the checkpoint: its weights are gated, so
# they are downloaded separately into pretrained/dinov3/ (see MODELS.md).
export DINOV3_WEIGHTS_DIR=${DINOV3_WEIGHTS_DIR:-pretrained/dinov3}

python main.py \
--model PINE-B/16 --feature_embedding_type dinov3_vitb16 \
--stream_attention isolated_instance --dino_grounding attention \
--camera_positional_encoding --reference_focal 420 \
--evaluate_gen --eval_wild_images --depth_colormap Spectral_r \
--eval_wild_images_dir ${IMAGE_DIR} \
--img_size 512 --num_sampling_steps 1 --noise_scale 2.0 --generate_noise_scale 2.0 \
--pretrained ${CHECKPOINT} --output_dir "${OUTPUT_DIR}"
