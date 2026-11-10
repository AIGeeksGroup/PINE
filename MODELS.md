# Model Zoo

- We assume that the downloaded weights are stored in the `pretrained` directory. It is recommended to create a
  symbolic link from `YOUR_MODEL_PATH` to `pretrained`:
```
ln -s YOUR_MODEL_PATH pretrained
```

- No checkpoint holds the frozen DINOv3 encoder: those weights are gated and cannot be redistributed here. They are
  downloaded once, separately, and attached at load time. See [DINOv3 encoder](#dinov3-encoder) below.


## PINE

PINE-B, PINE-L and PINE-H are trained at `512x512` on the six-dataset mixture
([dataloader/configs/six_datasets_512.yaml](dataloader/configs/six_datasets_512.yaml)) with
[scripts/train_b_512.sh](scripts/train_b_512.sh), [scripts/train_l_512.sh](scripts/train_l_512.sh) and
[scripts/train_h_512.sh](scripts/train_h_512.sh). Rel<sup>p</sup> is the point map error on the seven evaluation
datasets with one sampling step, with and without intrinsics.

| Model  | DINOv3 Encoder | Rel<sup>p</sup> with K | Rel<sup>p</sup> without K |
| ------ | :------------: | :--------------------: | :-----------------------: |
| PINE-B |    ViT-B/16    |          4.69          |           5.62            |
| PINE-L |    ViT-L/16    |          3.76          |           4.70            |
| PINE-H |   ViT-H+/16    |          3.33          |           4.16            |

A checkpoint is the `checkpoint-last.pth` that training writes: the weights (`model`), their EMA (`model_ema1`, which
evaluation and the demo use) and the training arguments (`args`).

The PINE weights are for non-commercial use only, under the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.


## PointDiT initialisation

Training starts from the released `512x512` PointDiT weights of the same size
([Hugging Face](https://huggingface.co/haofeixu/pointdit)). The training scripts expect these files in `pretrained/`:

| File                                        | Used by |
| ------------------------------------------- | :-----: |
| pointditb-512-mixdata-nodinov3-1d42aacf.pth | PINE-B  |
| pointditl-512-mixdata-nodinov3-240c1a4f.pth | PINE-L  |
| pointdith-512-mixdata-nodinov3-cb01dd3b.pth | PINE-H  |

```bash
pip install -U "huggingface_hub[cli]"
hf download haofeixu/pointdit pointditl-512-mixdata-nodinov3-240c1a4f.pth --local-dir pretrained
```

The point stream takes the PointDiT weights; the normal and the instance stream start as copies of its input, output
and time layers; the grounding and camera modules are new.


## DINOv3 encoder

PINE conditions on a frozen DINOv3 encoder. Request access on the
[official DINOv3 repository](https://github.com/facebookresearch/dinov3) and place the weights in
`pretrained/dinov3/`, under their exact upstream filenames:

```
pretrained/dinov3/
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth      # PINE-B
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth      # PINE-L
└── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth  # PINE-H
```

If they live somewhere else, point `DINOV3_WEIGHTS_DIR` at that directory instead:

```bash
export DINOV3_WEIGHTS_DIR=/path/to/dinov3
```

Every launch script in [scripts/](scripts) reads that variable and falls back to `pretrained/dinov3`.
