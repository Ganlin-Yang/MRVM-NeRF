
python run_training.py \
--base_data_dir /data/v-yangganlin/Neuray/data \
--save_dir ./data \
--cfg configs/gen/neuray_gen_depth.yaml \
--resume \
--pre_stageone \
--mask_ratio 0.5 \
--stage_two_begin 20000 \
--stage_two_warmup 10000 \
--lambda_feat 0.1
# replace 'depth' with 'cost_volume' for cost volume method

