# ScaFu: A Lightweight Scale Fusion Module for Transformer-based Object Detection

ScaFu is a lightweight yet powerful scale fusion module designed to enhance Transformer-based object detection models. It leverages multi-scale feature extraction, cross-scale interaction, and adaptive detail enhancement to significantly improve spatial detail perception in detection tasks.

---

## 1. Environment Setup

To set up the environment, you can use conda to create a new environment with Python 3.11 and install required dependencies using the following commands:

```bash
conda create -n scafu python=3.11 -y
conda activate scafu
pip install -r requirements.txt
```

---

## 2. Data Preparation

Before training, you need to modify the paths in `coco_detection.yml` to point to the correct locations of the COCO dataset. Specifically, you need to update the `train_dataloader` and `val_dataloader` paths:

```yaml
train_dataloader:
  img_folder: /data/COCO2017/train2017/
  ann_file: /data/COCO2017/annotations/instances_train2017.json

val_dataloader:
  img_folder: /data/COCO2017/val2017/
  ann_file: /data/COCO2017/annotations/instances_val2017.json
```

---

## 3. Backbone Preparation

We use **DINOv3-S+** as the backbone. You can download the pre-trained models from the [DINOv3 repository](https://github.com/facebookresearch/dinov3).

Once you have downloaded the models, place them in the `./ckpts` folder as follows:

```bash
ckpts/
 ├── dinov3_vits16.pth
```

---

## 4. Training

To train the model, run the following command for ViT-based variants. Make sure you have the correct GPU setup and `torchrun` installed.

```bash
# for ViT-based variants
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/scafu/scafu_dinov3_x_coco.yml --use-amp --seed=3407
```

---

## 5. Acknowledgement

Our work is built upon the following projects:

* [D-FINE](https://github.com/Peterande/D-FINE)
* [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
* [DEIM](https://github.com/ShihuaHuang95/DEIM)
* [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
* [DINOv3](https://github.com/facebookresearch/dinov3)

Thanks for their great work!

---




