#!/bin/bash
# category-agnostic ShapeNet-all
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


# category-agnostic ShapeNet-unseen
python train/train.py \
-c conf/exp/sn64_unseen.conf \
--resume \
--gpu_id 0,1,2,3 \
-n sn64_unseen \
--logs_path /output/logs \
--checkpoints_path /output/checkpoints \
--visual_path /output/visuals \
--itera 300000 \
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


# category-specific ShapeNet-chair/ShapeNet-car
python train/train.py \
-c conf/exp/srn.conf \
--resume \
--gpu_id 0,1,2,3,4,5,6,7 \
-n srn_chair \
--logs_path /output/logs \
--checkpoints_path /output/checkpoints \
--visual_path /output/visuals \
--itera 400000 \
--lr 2e-4 \
--warm_up_iter 5000 \
-D /data/nerformer_data/chairs \
--ray_batch_size 256 \
--stage pretrain \
--mask_ratio 0.5 \
--lambda_recons 0.1 \
--stage_two_begin 20000 \
--stage_two_warmup 5000 \
--EMA 0.99 \
-B 8 \
-V 1,2 \
--no_bbox_step 300000 \
# replace "chair" with "car" for ShapeNet-car setting


