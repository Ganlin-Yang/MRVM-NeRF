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

# category-agnostic ShapeNet-unseen
python train/train.py \
-c conf/exp/sn64_unseen.conf \
--resume \
--gpu_id 0,1,2,3 \
-n sn64_unseen \
--logs_path /output/logs \
--checkpoints_path /output/checkpoints \
--visual_path /output/visuals \
--itera 600000 \
--lr 1e-3 \
--warm_up_iter 0 \
-D /data/nerformer_data/NMR_Dataset \
--ray_batch_size 128 \
--stage fine_tune \
--load_iter 300000 \
--path_to_load /output/checkpoints/sn64_unseen_prt \
-B 16 \
-V 1 \
--no_bbox_step 0 \

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
--lr 1e-3 \
--warm_up_iter 0 \
-D /data/nerformer_data/chairs \
--ray_batch_size 256 \
--stage fine_tune \
--load_iter 400000 \
--path_to_load /output/checkpoints/srn_chair_prt \
-B 8 \
-V 1,2 \
--no_bbox_step 0 \
# replace "chair" with "car" for ShapeNet-car setting