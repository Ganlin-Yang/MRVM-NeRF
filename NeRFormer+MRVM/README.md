
## Usage
### Environment Setup
```shell
conda env create -f environment.yaml
conda activate nerformer_mrvm
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Data preparation
```shell
pip install gdown
mkdir /data/nerformer_data
cd /data/nerformer_data
wget https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
unzip NMR_Dataset.zip
cd ./NMR_Dataset
gdown --id 1Uxf0GguAUTSFIDD_7zuPbxk1C9WgXjce # download genlist.py
python genlist.py
cd /data/nerformer_data
gdown --id 19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU
gdown --id 1VWtRZkC4_ON6EBhNNBTag-BwsjPB2Yc8
unzip srn_chairs.zip
unzip srn_cars.zip
mv chairs_train/chairs_2.0_train ./
rm -rf chairs_train
mv chairs_2.0_train/ chairs_train/
```
The processed dataset should be organized as:
```shell
    data
      |--nerformer_data
          |-- cars_test
          |-- cars_train
          |-- cars_val
          |-- chairs_test
          |-- chairs_train
          |-- chairs_val
          |-- NMR_Dataset
               |-- 02691156
               |-- genlist.py
               |-- metadata.yaml
```
    
3. Download [pretrained-models](https://drive.google.com/drive/folders/16vpZwuKuK3NkHwir3zM_lyMdxbFFZ70z?usp=sharing) and organize as:
    ```shell
    data
      |--model
          |-- sn64
               |-- net_800000
          |-- sn64_unseen
               |-- net_600000
          |-- srn_car
               |-- net_400000
          |-- srn_chair
               |-- net_400000
          
    ```

## Evaluation
```shell
## Evaluation on category-agnostic ShapeNet-all setting
python eval/eval.py \
-c conf/exp/sn64.conf \
--gpu_id 0,1,2,3 \
-n sn64 \
-D /data/nerformer_data/NMR_Dataset \
--load_iter 800000 \
--path_to_load /data/model/sn64 \
--multicat \
-L viewlist/src_dvr.txt \
-O ./eval_out \
```
For more details, please refer to [scripts/eval.sh](./scripts/eval.sh).


## Generalizable model mask-based pretraining

```shell
## category-agnostic ShapeNet-all setting mask-based pretraining
python train/train.py \
-c conf/exp/sn64.conf \
--resume \
--gpu_id 0,1,2,3 \
-n sn64 \
--logs_path /output/logs \
--checkpoints_path /output/checkpoints \
--visual_path /output/visuals \
--itera 400000 \
--lr 2e-4 \
--warm_up_iter 5000 \
-D /data/nerformer_data/NMR_Dataset \
--ray_batch_size 128 \
--stage pretrain \
--mask_ratio 0.5 \
--lambda_recons 0.1 \
--stage_two_begin 20000 \
--stage_two_warmup 5000 \
--EMA 0.99 \
-B 16 \
-V 1 \
--no_bbox_step 300000 \
```
For more details, please refer to [scripts/pretrain.sh](./scripts/pretrain.sh).


## Generalizable model finetuning without masking
```shell
# category-agnostic ShapeNet-all setting finetuning without masking
python train/train.py \
-c conf/exp/sn64.conf \
--resume \
--gpu_id 0,1,2,3 \
-n sn64 \
--logs_path /output/logs \
--checkpoints_path /output/checkpoints \
--visual_path /output/visuals \
--itera 800000 \
--lr 1e-3 \
--warm_up_iter 0 \
-D /data/nerformer_data/NMR_Dataset \
--ray_batch_size 128 \
--stage fine_tune \
--load_iter 400000 \
--path_to_load /output/checkpoints/sn64_prt \
-B 16 \
-V 1 \
--no_bbox_step 0 \
```

For more details, please refer to [scripts/finetune.sh](./scripts/finetune.sh).



## Acknowledgements
In this repository, we have used codes or datasets from the following repositories. 
We thank all the authors for sharing great codes or datasets.

- [NeRF-official](https://github.com/bmild/nerf) and [NeRF-torch](https://github.com/yenchenlin/nerf-pytorch)
- [PixelNeRF](https://github.com/sxyu/pixel-nerf)
- [DVR](https://github.com/autonomousvision/differentiable_volumetric_rendering)
- [co3d](https://github.com/facebookresearch/co3d?tab=readme-ov-file)
- [GNT](https://github.com/VITA-Group/GNT)
## Citation
```
@inproceedings{yang2024maskbased,
title={Mask-based modeling for Neural Radiance Fields},
author={Ganlin Yang and Guoqiang Wei and Zhizheng Zhang and Yan Lu and Dong Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=SEiuSzlD1d}
}

@inproceedings{yu2021pixelnerf,
      title={{pixelNeRF}: Neural Radiance Fields from One or Few Images},
      author={Alex Yu and Vickie Ye and Matthew Tancik and Angjoo Kanazawa},
      year={2021},
      booktitle={CVPR},
}

@inproceedings{reizenstein2021common,
  title={Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction},
  author={Reizenstein, Jeremy and Shapovalov, Roman and Henzler, Philipp and Sbordone, Luca and Labatut, Patrick and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10901--10911},
  year={2021}
}
```
