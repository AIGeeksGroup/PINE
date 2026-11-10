# Datasets

All paths are relative to the repository root. `datasets` is a symlink to the folder that holds the data:

```
datasets/
├── train/   # the training sets of dataloader/configs/six_datasets_512.yaml
└── eval/    # the evaluation benchmarks
```

## Training datasets

The released models train on four synthetic datasets with geometry and two real datasets with instance labels
([dataloader/configs/six_datasets_512.yaml](dataloader/configs/six_datasets_512.yaml), 162,632 samples per epoch).
Every dataset is a folder of HDF5 shards that hold the released image bytes and labels.

| Dataset | Supervises | Samples | Folder | Preparation |
|---|---|---:|---|---|
| Hypersim | point, normal, instance | 39,241 | `datasets/train/hypersim` | `scripts/train_prepare_hypersim_h5.py` |
| Virtual KITTI 2 | point, normal, instance | 20,148 | `datasets/train/vkitti2` | `scripts/train_prepare_vkitti2_h5.py` |
| UrbanSyn | point, normal, instance | 7,539 | `datasets/train/urbansyn_h5` | `scripts/train_prepare_urbansyn_download.py`, `scripts/train_prepare_urbansyn_h5.py` |
| Structured3D | point, normal | 72,058 | `datasets/train/structured3d_h5` | `scripts/train_prepare_structured3d_download.sh`, `scripts/train_prepare_structured3d_h5.py` |
| EntitySeg | instance | 6,369 | `datasets/train/entityseg_things_h5` | `scripts/train_prepare_entityseg_download.py`, `scripts/train_prepare_entityseg_h5.py` |
| ADE20K | instance | 17,277 | `datasets/train/ade20k_h5` | `scripts/train_prepare_ade20k_download.sh`, `scripts/train_prepare_ade20k_h5.py` |

Each preparation script documents its inputs and its label conventions in its header. These commands pack the sets of
the released models (`RAW` is a folder for the downloads):

```bash
out=datasets/train
cameras=datasets/train/camera_metadata

# Hypersim
python scripts/train_prepare_hypersim_h5.py --root RAW/hypersim --workers 16 \
    --camera-parameters RAW/metadata_camera_parameters.csv --output-dir $out/hypersim --camera-dir $cameras
python scripts/train_prepare_hypersim_instance_counts.py --root $out/hypersim \
    --output $out/hypersim_instance_counts.json

# Virtual KITTI 2
python scripts/train_prepare_vkitti2_h5.py --root RAW/vkitti2 --output-dir $out/vkitti2 --camera-dir $cameras

# UrbanSyn (the second command writes the camera sidecar)
python scripts/train_prepare_urbansyn_download.py --output-dir RAW/urbansyn
urbansyn="--root RAW/urbansyn --output-dir $out/urbansyn_h5 --camera-dir $cameras --stats-dir RAW/urbansyn_stats"
python scripts/train_prepare_urbansyn_h5.py $urbansyn
python scripts/train_prepare_urbansyn_h5.py $urbansyn --write-sidecar

# Structured3D (zip 09 is broken upstream; zip 17 holds test scenes only)
bash scripts/train_prepare_structured3d_download.sh RAW/structured3d 8 00 01 02 03 04 05 06 07 08 10 11 12 13 14 15 16
curl -o RAW/structured3d/Structured3D_bbox.zip \
    https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/Structured3D_bbox.zip
structured3d="--root RAW/structured3d --scene-limit 3250 --worker 0 --workers 1 --output-dir $out/structured3d_h5"
structured3d="$structured3d --camera-dir $cameras --stats-dir RAW/structured3d_stats"
python scripts/train_prepare_structured3d_h5.py $structured3d
python scripts/train_prepare_structured3d_h5.py $structured3d --write-sidecar

# EntitySeg (accept the terms of qqlu1992/Adobe_EntitySeg on Hugging Face first)
python scripts/train_prepare_entityseg_download.py --output-dir RAW/entityseg --image-zips \
    images_lr/entity_01_11580.zip images_lr/entity_02_11598.zip images_lr/entity_03_10049.zip
python scripts/train_prepare_entityseg_h5.py --root RAW/entityseg --exclude-sources coco_val --shard-size 1000 \
    --things-only --worker 0 --workers 1 --output-dir $out/entityseg_things_h5 --as-things door window glass_window \
    stair step banister fireplace countertop bar stove potted_plants christmas_tree pipe ventilation escalator \
    seating_area cage balcony fence wiredfence pole electricpole palm awning fountain

# ADE20K
bash scripts/train_prepare_ade20k_download.sh RAW/ade20k
python scripts/train_prepare_ade20k_h5.py --root RAW/ade20k --shard-size 1000 --worker 0 --workers 1 \
    --output-dir $out/ade20k_h5 --ignore plant rug base stairway blind bar escalator poster buffet "conveyer belt" \
    cradle tank blanket shower "crt screen"
```

`--worker k --workers n` splits a packing job over n processes; the shards do not depend on n.

- **Hypersim.** Download the scenes with the tools of [ml-hypersim](https://github.com/apple/ml-hypersim): the
  `tonemap.jpg` previews and the `depth_meters`, `normal_cam` and `semantic_instance` arrays. The packer also needs
  `contrib/mikeroberts3000/metadata_camera_parameters.csv` of that repository. It packs the 39,241 training frames of
  `dataloader/configs/hypersim_train_frames.txt`.
- **Virtual KITTI 2.** Download the rgb, depth, instanceSegmentation and textgt archives of
  [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/). The dataset has no normals: make
  them from the depth with `utils/depth2normal.sh` of [Lotus](https://github.com/EnVision-Research/Lotus).
- **Hypersim instance filter.** Frames without an instance and scenes with fewer than 10 labelled objects get no
  instance loss (`hypersim_instance_counts.json`).
- **Camera sidecars.** The packing scripts of Hypersim, Virtual KITTI 2, UrbanSyn and Structured3D write a camera
  sidecar to `--camera-dir`. The training scripts read the sidecars from `CAMERA_METADATA_ROOT` (default
  `datasets/train/camera_metadata`).

## Evaluation datasets

### Point map and depth (7 datasets)

All are real captures, disjoint from training. "Boundary" marks the datasets on which the scale-invariant boundary F1
is also reported.

| Dataset | Domain | #Samples | Boundary | Download |
|---|---|---:|:---:|---|
| DIODE | indoor + outdoor | 771 | | [DIODE.zip](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/DIODE.zip) |
| KITTI | outdoor (driving) | 652 | | [KITTI.zip](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/KITTI.zip) |
| NYUv2 | indoor | 654 | | [NYUv2.zip](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/NYUv2.zip) |
| ETH3D | indoor + outdoor | 454 | | [ETH3D.zip](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/ETH3D.zip) |
| HAMMER | indoor (multi-modal) | 775 | ✓ | [HAMMER.zip](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/HAMMER.zip) |
| iBims-1 | indoor | 100 | ✓ | [iBims-1.zip](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/iBims-1.zip) |
| Booster | indoor (transparent/specular) | 38 | ✓ | [Booster.zip](https://huggingface.co/datasets/haofeixu/pointdit-eval/resolve/main/Booster.zip) |
| **Total** | | **3,444** | | |

**All 7 sets are available pre-processed**, one zip per dataset, 2.9 GB total, split over
two Hugging Face repositories:

> Six of them from **[Ruicheng/monocular-geometry-evaluation](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation)**,
> the evaluation sets released with MoGe, and Booster from
> **[haofeixu/pointdit-eval](https://huggingface.co/datasets/haofeixu/pointdit-eval)**.

Nothing needs converting. Download the zips into `datasets/eval/` and unpack:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p datasets/eval

hf download Ruicheng/monocular-geometry-evaluation --repo-type dataset \
  --local-dir datasets/eval DIODE.zip KITTI.zip NYUv2.zip ETH3D.zip HAMMER.zip iBims-1.zip

hf download haofeixu/pointdit-eval --repo-type dataset \
  --local-dir datasets/eval Booster.zip

cd datasets/eval && unzip '*.zip'   # one zip per dataset
```

### Normals (4 datasets)

NYUv2, ScanNet, iBims-1 and Sintel with the full-frame protocol of [DSINE](https://github.com/baegwangbin/DSINE).
Download the DSINE evaluation data and its split lists, and set

```bash
export DSINE_DATA_ROOT=datasets/eval/normal          # nyuv2/, scannet/, ibims/, sintel/
export DSINE_SPLIT_ROOT=datasets/eval/normal_splits  # the split lists of the DSINE repository
```

### Instances (5 benchmarks)

Point-prompted instance segmentation under the [gen2seg](https://github.com/UCDvision/gen2seg) protocol, in
`datasets/eval/instance_segmentation`:

```
instance_segmentation/
├── coco/     # annotations/instances_val2017.json, val2017/
├── dram/     # test/dram.txt, test/<name>.jpg, labels/<name>.png
├── egohos/   # test_indomain/{image,label}/, test_outdomain/{image,label}/
├── ishape/   # <antenna|branch|fence|hanger|log|wire>/val/{image,instance_map}/
└── pidray/   # annotations/test_easy.json, test/
```

### Planarity

`scripts/eval_ibims_planarity.py` needs the official `ibims1_core_mat` files, their image list and the official
evaluation scripts of [iBims-1](https://www.asg.ed.tum.de/lmf/ibims1/).

## Dataset mixture config format

```yaml
img_size: 512
samples_per_epoch: 160000        # optional; the default is the sum of the dataset sizes
datasets:
  - name: hypersim_h5            # loader name (dataloader/h5_shards.py)
    data_path: datasets/train/hypersim
    weight: 0.198                # share of the samples of an epoch
    max_depth: 30.0              # depths above it are invalid, or sky with handle_sky
    load_instance: true          # also load instance ids
  - name: entityseg_h5
    task: instance_segmentation  # an image set with instance labels only
    data_path: datasets/train/entityseg_things_h5
    weight: 0.108
    instance_background_supervised: true
```

The weights set the sampling share of each dataset, whatever its size (`MixedImageDepthIntrinsicsDataset` builds a
weighted sampler).
