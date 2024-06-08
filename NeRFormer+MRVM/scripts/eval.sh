#!/bin/bash
# category-agnostic ShapeNet-all
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


# category-agnostic ShapeNet-unseen
python eval/eval.py \
-c conf/exp/sn64_unseen.conf \
--gpu_id 0,1,2,3 \
-n sn64_unseen \
-D /data/nerformer_data/NMR_Dataset \
--load_iter 600000 \
--path_to_load /data/model/sn64_unseen \
--multicat \
-L viewlist/src_gen.txt \
-O ./eval_out \


# category-specific ShapeNet-chair/ShapeNet-car 1 source view
python eval/eval.py \
-c conf/exp/srn.conf \
--gpu_id 0,1,2,3 \
-n srn_chair \
-D /data/nerformer_data/chairs \
--load_iter 400000 \
--path_to_load /data/model/srn_chair \
-P 64 \
-O ./eval_out \
# replace "chair" with "car" for ShapeNet-car setting

# category-specific ShapeNet-chair/ShapeNet-car 2 source views
python eval/eval.py \
-c conf/exp/srn.conf \
--gpu_id 0,1,2,3,4,5,6,7 \
-n srn_chair \
-D /data/nerformer_data/chairs \
--load_iter 400000 \
--path_to_load /data/model/srn_chair \
-P 64,104 \
-O ./eval_out \
# replace "chair" with "car" for ShapeNet-car setting


