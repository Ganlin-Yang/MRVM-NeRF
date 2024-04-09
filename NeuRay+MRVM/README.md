
## Usage
### Environment Setup
```shell
conda env create -f environment.yaml
conda activate neuray_mrvm
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Data preparation
1. Download training datasets: | [Google Scanned Objects](https://github.com/googleinterns/IBRNet#e-google-scanned-objects) | [RealEstate10K](https://github.com/googleinterns/IBRNet#d-realestate10k) | [LLFF released Scenes](https://github.com/googleinterns/IBRNet#b-llff-released-scenes) |
[Space Dataset](https://github.com/googleinterns/IBRNet#c-spaces-dataset)  | [DTU-Train](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/EXcPUeyIqAdHrS2LUCmrRJwB8UN0QItiPBm90YuldNm0Ig?e=2POyCI) | [Colmap depth for forward-facing scene](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/EX0M0c_DyUFDiz1c-ebSO_oBTEeWk8jRYNwCHMgbFH0Pww?e=bO9stn) | [Colmap depth for DTU](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/EfkjOG2b1epNl322dE3EOeQBAm_Ncver5EmPN4mOZE0ZnA?e=R975nx) |

   The training datasets should be organized as follows:
   ```shell
   data
    |-- google_scanned_objects
    |-- real_estate_dataset   
    |-- real_iconic_noface
    |-- spaces_dataset
    |-- dtu_train
    |-- colmap_forward_cache
    |-- colmap_dtu_cache
   ```

2. Download testing datasets: | [DTU-Test](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/ESZ5vNtkX6dJlJKt_xoJXkMBwLHmPvnXF0UQhaJQIw858w?e=u2DqHd) | [LLFF](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/EbI1OMqOjOdEtS3NqNguPXsBXOfEnG0MWMmD0If-7OR4dg?e=bf6Pvu) | [NeRF Synthetic](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/yuanly_connect_hku_hk/Ec7yNxwmVbBDmccPar34yOgBwGDyztVfpV-XRIhyKLEg2Q?e=gYKSTm) |

    The testng datasets should be organized as follows:
   ```shell
   data
    |-- dtu_test
    |-- llff_colmap
    |-- nerf_synthetic
   ```
3. Download [pretrained-models](https://drive.google.com/drive/folders/16vpZwuKuK3NkHwir3zM_lyMdxbFFZ70z?usp=sharing) and organize as:
    ```shell
    data
      |--model
          |-- generalization
               |-- cost_volume.pth
               |-- depth.pth
          |-- finetune
               |-- DTU_test
                      |-- birds.pth
               |-- LLFF_test
                      |-- fern.pth
               |-- Synthetic_test
                      |-- drums.pth
    ```

## Evaluation
```shell
## render on snowman of the DTU dataset in cross-scene generalization setting
python render.py --database_name dtu_test/snowman/black_800 \
                 --cfg configs/gen/neuray_gen_depth.yaml \
                 --render_type gen \
                 --base_data_dir /data/v-yangganlin/Neuray/data \ # your data path
                 --save_dir ./data \  # your save path
                 --load_ckpt_path ./data/model/generalization/depth.pth # your ckpt path

```
For more details, please refer to [scripts/eval.sh](./scripts/eval.sh).


## Cross-scene generalizable model training

### Mask-based pretraining across scenes
 
```shell
python run_training.py \
--base_data_dir /data/v-yangganlin/Neuray/data \ # your data path
--save_dir ./data \ # your save path
--cfg configs/gen/neuray_gen_depth.yaml \ 
--pre_stageone \ # indicate mask-based pretraining stage
--mask_ratio 0.5 \ # mask ratio
--stage_two_begin 20000 \ # starting iteration of adding mask-prediction task as an auxiliary task 
--stage_two_warmup 10000 \ # the lasting iterations of adding mask-prediction task gradually like warmup
--lambda_feat 0.1 # the loss weight for mask-prediction task
```
Please refer to [scripts/cross_scene_pretrain.sh](./scripts/cross_scene_pretrain.sh).

### Finetuning without masking across scenes

```shell
python run_training.py \
--base_data_dir /data/v-yangganlin/Neuray/data \ # your data path
--save_dir ./data \ # your save path
--cfg configs/gen/neuray_gen_depth.yaml \ 
--ft_load_path ./data/model/neuray_gen_depth_train_pre/model.pth  # mask-based pretraining ckpt
```

Please refer to [scripts/cross_scene_finetune.sh](./scripts/cross_scene_finetune.sh).

## Per-scene specific model finetuning
```shell
# per-scene finetune on birds in the DTU dataset
python run_training.py \
--base_data_dir /data/v-yangganlin/Neuray/data \ # your data path
--save_dir ./data \ # your save path
--cfg configs/ft/depth/dtu/neuray_ft_depth_birds.yaml \
--ft_load_path ./data/model/neuray_gen_depth_train_ft/model.pth # cross-scene generalization ckpt
```

For more details, please refer to [scripts/per_scene_finetune.sh](./scripts/per_scene_finetune.sh).



## Acknowledgements
In this repository, we have used codes or datasets from the following repositories. 
We thank all the authors for sharing great codes or datasets.
- [NeuRay](https://github.com/liuyuan-pal/NeuRay)
- [IBRNet](https://github.com/googleinterns/IBRNet)
- [MVSNet-official](https://github.com/YoYo000/MVSNet) and [MVSNet-kwea123](https://github.com/kwea123/CasMVSNet_pl)
- [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
- [NeRF-official](https://github.com/bmild/nerf) and [NeRF-torch](https://github.com/yenchenlin/nerf-pytorch)
- [MVSNeRF](https://github.com/apchenstu/mvsnerf)
- [PixelNeRF](https://github.com/sxyu/pixel-nerf)
- [COLMAP](https://github.com/colmap/colmap)
- [IDR](https://lioryariv.github.io/idr/)
- [RealEstate10K](https://google.github.io/realestate10k/)
- [DeepView](https://augmentedperception.github.io/deepview/)
- [Google Scanned Objects](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)
- [LLFF](https://github.com/Fyusion/LLFF)
- [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36)

## Citation
```
@inproceedings{
yang2024maskbased,
title={Mask-based modeling for Neural Radiance Fields},
author={Ganlin Yang and Guoqiang Wei and Zhizheng Zhang and Yan Lu and Dong Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=SEiuSzlD1d}
}

@inproceedings{liu2022neuray,
  title={Neural Rays for Occlusion-aware Image-based Rendering},
  author={Liu, Yuan and Peng, Sida and Liu, Lingjie and Wang, Qianqian and Wang, Peng and Theobalt, Christian and Zhou, Xiaowei and Wang, Wenping},
  booktitle={CVPR},
  year={2022}
}
```
