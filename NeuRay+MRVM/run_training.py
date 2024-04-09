import argparse
from globals import *
parser = argparse.ArgumentParser()

parser.add_argument('--base_data_dir', type=str, default='/data/v-yangganlin/Neuray/data')
parser.add_argument('--save_dir', type=str, default='./data')

# command line control paras which are directly written into yaml file
parser.add_argument('--cfg', type=str, default='configs/gen/neuray_gen_depth.yaml')
parser.add_argument('--ft_load_path', type=str, default=None)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--pre_stageone', action='store_true')
parser.add_argument('--mask_ratio', type=float, default=0.5)
parser.add_argument('--stage_two_begin', type=int, default=1)
parser.add_argument('--stage_two_warmup', type=int, default=3)
parser.add_argument('--lambda_rgb', type=float, default=1.0)
parser.add_argument('--lambda_feat', type=float, default=0.1)
parser.add_argument('--lambda_consist', type=float, default=1.0)
parser.add_argument('--lambda_depth', type=float, default=1.0)
parser.add_argument('--total_step', type=int, default=-1)
parser.add_argument('--extra_save_interval', type=int, default=-1)
flags = parser.parse_args()

base_data_dir_lst.append(flags.base_data_dir)
save_dir_lst.append(flags.save_dir)
from train.trainer import Trainer
from utils.base_utils import load_cfg

origin_cfg = load_cfg(flags.cfg)

if origin_cfg['network'] == 'neuray_gen' and flags.pre_stageone:
    origin_cfg['stage'] = 'pretrain'
    origin_cfg['ft_load_path'] = None
    origin_cfg['mask_ratio'] = flags.mask_ratio
    origin_cfg['stage_two_begin'] = flags.stage_two_begin
    origin_cfg['stage_two_warmup'] = flags.stage_two_warmup
    origin_cfg['lambda_feat'] = flags.lambda_feat
    origin_cfg['name'] = origin_cfg['name'] + '_pre' 

else: # cross scene unmask ft or per-scene unmask ft
    origin_cfg['stage'] = 'ft'
    origin_cfg['ft_load_path'] = flags.ft_load_path 
    origin_cfg['mask_ratio'] =  0.0
    origin_cfg['lambda_feat'] = 0.0
    origin_cfg['name'] = origin_cfg['name'] + '_ft'

origin_cfg['ray_batch_num'] = 4096
origin_cfg['lambda_rgb'] = flags.lambda_rgb
origin_cfg['lambda_consist'] = flags.lambda_consist
origin_cfg['lambda_depth'] = flags.lambda_depth
if flags.total_step > 0:
   origin_cfg['total_step'] = flags.total_step
if flags.extra_save_interval > 0:
   origin_cfg['extra_save_interval'] = flags.extra_save_interval
trainer = Trainer(origin_cfg)
trainer.run(resume=flags.resume)