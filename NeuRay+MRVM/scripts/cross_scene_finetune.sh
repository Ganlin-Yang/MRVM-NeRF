
python run_training.py \
--base_data_dir /data/v-yangganlin/Neuray/data \
--save_dir ./data \
--cfg configs/gen/neuray_gen_depth.yaml \
--ft_load_path ./data/model/neuray_gen_depth_train_pre/model.pth \
--resume
# for cost volume method, replace 'depth' with 'cost_volume'

