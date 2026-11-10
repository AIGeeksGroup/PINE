#!/usr/bin/env bash
# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.

# PINE-H at 512x512 on the six-dataset mixture, from the released PointDiT-H weights (see MODELS.md).
# Original run: 8 GPUs x batch 4 x 2 accumulation step(s) (effective batch 64), lr 1e-5, 20 epochs of 162,632
# samples. Keep GPUs x --batch_size x --accum_iter at 64.
# Every batch has one frame size: width / height log-uniform in [0.5, 3.5] at 1024 patches, crops of side 0.6-1.0 of
# the largest crop of that aspect ratio (dataloader/aspect_batches.py).
# --online_eval scores every sample of the seven point map sets, the four normal sets, COCOexc and DRAM after each
# epoch, with the dataset intrinsics and without a camera.

cd "$(dirname "$0")/.."

# Multi-node: raise --num_machines, set --machine_rank per node, and add
# --main_process_ip <rank-0 host> --main_process_port 29500 to the launcher.

DATASET_CONFIG=dataloader/configs/six_datasets_512.yaml
EVAL_DATA=datasets/eval
PRETRAINED=pretrained/pointdith-512-mixdata-nodinov3-cb01dd3b.pth
OUTPUT_DIR=checkpoints/pine-h-512
export DINOV3_WEIGHTS_DIR=${DINOV3_WEIGHTS_DIR:-pretrained/dinov3}
# The DSINE normal benchmarks and their split lists (see DATASETS.md).
export DSINE_DATA_ROOT=${DSINE_DATA_ROOT:-${EVAL_DATA}/normal}
export DSINE_SPLIT_ROOT=${DSINE_SPLIT_ROOT:-${EVAL_DATA}/normal_splits}
# The camera sidecars of the packed training sets (see DATASETS.md).
export CAMERA_METADATA_ROOT=${CAMERA_METADATA_ROOT:-datasets/train/camera_metadata}
export OPENCV_IO_ENABLE_OPENEXR=1
mkdir -p ${OUTPUT_DIR}

accelerate launch --num_processes 8 --num_machines 1 --machine_rank 0 \
--mixed_precision bf16 --dynamo_backend no \
main.py \
--model PINE-H/16 --feature_embedding_type dinov3_vith16plus \
--stream_attention isolated_instance --dino_grounding attention \
--camera_positional_encoding --reference_focal 420 --camera_true_prob 0.8 \
--split all \
--img_size 512 --noise_scale 2.0 \
--dataset_config ${DATASET_CONFIG} \
--random_aspect_range 0.5 3.5 --frame_side_range 0.6 1.0 \
--P_mean -0.8 --P_std 0.8 --force_zero_t_ratio 0.1 --t_eps 0.05 \
--sky_loss_weight 0.01 --noise_fill_invalid \
--rel_point_loss_weight 0.1 --rel_point_loss_exclude_sky --gradient_loss_weight 0.5 \
--normal_loss_weight 1 --normal_sink_loss_weight 1.0 --sink_window 5 --sink_iters 5 --sink_tau 0.1 \
--instance_loss_weight 0.02 \
--batch_size 4 --accum_iter 2 --lr 1e-5 --lr_schedule constant --weight_decay 0 --grad_clip \
--epochs 20 --warmup_epochs 0 --ema_decay1 0.999 --seed 0 --num_workers 8 \
--save_last_freq 1 \
--online_eval --eval_freq 1 --num_sampling_steps 1 --generate_noise_scale 2.0 --eval_no_save_gen \
--eval_depth_dataset --eval_depth_data_root ${EVAL_DATA} \
--eval_depth_dataset_name DIODE,KITTI,NYUv2,ETH3D,HAMMER,iBims-1,Booster \
--eval_boundary_datasets HAMMER,iBims-1,Booster \
--eval_depth_resize_height 512 \
--eval_normal_benchmarks nyuv2,scannet,ibims,sintel \
--eval_instance_root ${EVAL_DATA}/instance_segmentation \
--pretrained ${PRETRAINED} --pretrained_weights ema \
--output_dir ${OUTPUT_DIR} --resume \
2>&1 | tee -a ${OUTPUT_DIR}/train.log
