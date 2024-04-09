# per-scene finetune on birds in the DTU dataset
python run_training.py \
--base_data_dir /data/v-yangganlin/Neuray/data \
--save_dir ./data \
--cfg configs/ft/depth/dtu/neuray_ft_depth_birds.yaml \
--ft_load_path ./data/model/neuray_gen_depth_train_ft/model.pth \
--resume


# per-scene finetune on fern in the LLFF dataset
python run_training.py \
--base_data_dir /data/v-yangganlin/Neuray/data \
--save_dir ./data \
--cfg configs/ft/depth/llff/neuray_ft_depth_fern.yaml \
--ft_load_path ./data/model/neuray_gen_depth_train_ft/model.pth \
--resume


# per-scene finetune on lego in the Synthetic dataset
python run_training.py \
--base_data_dir /data/v-yangganlin/Neuray/data \
--save_dir ./data \
--cfg configs/ft/depth/nerf_synthetic/neuray_ft_depth_lego.yaml \
--ft_load_path ./data/model/neuray_gen_depth_train_pre/model.pth \
--resume

# for finetuning on cost-volume method, replace 'depth' with 'cv' respectively.