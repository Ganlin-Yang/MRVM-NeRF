# The following represents evaluation under cross-scene generalization setting
## render on lego of the NeRF synthetic dataset
python render.py --database_name nerf_synthetic/lego/black_800 \
                 --cfg configs/gen/neuray_gen_depth.yaml \
                 --render_type gen \
                 --base_data_dir /data/v-yangganlin/Neuray/data \
                 --save_dir ./data \
                 --load_ckpt_path /data/v-yangganlin/model/neuray/generalization/depth.pth

## render on snowman of the DTU dataset
python render.py --database_name dtu_test/snowman/black_800 \
                 --cfg configs/gen/neuray_gen_depth.yaml \
                 --render_type gen \
                 --base_data_dir /data/v-yangganlin/Neuray/data \
                 --save_dir ./data \
                 --load_ckpt_path /data/v-yangganlin/model/neuray/generalization/depth.pth

## render on fern of the LLFF dataset
python render.py --database_name llff_colmap/fern/high \
                 --cfg configs/gen/neuray_gen_depth.yaml \
                 --render_type gen \
                 --base_data_dir /data/v-yangganlin/Neuray/data \
                 --save_dir ./data \
                 --load_ckpt_path /data/v-yangganlin/model/neuray/generalization/depth.pth

###########################################################################

# The following represents evaluation under per-scene finetuning setting
## render on lego of the NeRF synthetic dataset
python render.py --cfg configs/ft/depth/nerf_synthetic/neuray_ft_depth_lego.yaml \
                 --render_type ft  \
                 --base_data_dir /data/v-yangganlin/Neuray/data \
                 --save_dir ./data \
                 --load_ckpt_path /data/v-yangganlin/model/neuray/finetune/Synthetic_test/lego.pth

## render on snowman of the DTU dataset
python render.py --cfg configs/ft/depth/dtu/neuray_ft_depth_snowman.yaml \
                 --render_type ft \
                 --base_data_dir /data/v-yangganlin/Neuray/data \
                 --save_dir ./data \
                 --load_ckpt_path /data/v-yangganlin/model/neuray/finetune/DTU_test/snowman.pth
                 
## render on fern of the LLFF dataset
python render.py --cfg configs/ft/depth/llff/neuray_ft_depth_fern.yaml \
                 --render_type ft \
                 --base_data_dir /data/v-yangganlin/Neuray/data \
                 --save_dir ./data \
                 --load_ckpt_path /data/v-yangganlin/model/neuray/finetune/LLFF_test/fern.pth

# for evaluation on cost-volume method, replace 'depth' with 'cv' respectively.
