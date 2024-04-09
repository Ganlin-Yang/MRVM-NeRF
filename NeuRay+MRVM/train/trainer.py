import os

import torch
import numpy as np
from torch.nn import DataParallel
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from dataset.name2dataset import name2dataset
from network.loss import name2loss
from network.renderer import name2network
from train.lr_common_manager import name2lr_manager
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger, reset_learning_rate, MultiGPUWrapper, DummyLoss
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import simple_collate_fn, dummy_collate_fn
from globals import save_dir_lst

save_dir = save_dir_lst[0]
class Trainer:
    default_cfg={
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg":{
            "lr_init": 1.0e-4,
            "decay_step": 100000,
            "decay_rate": 0.5,
        },
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "extra_save_interval": 50000,
        "worker_num": 4,
    }
    def _init_dataset(self):
        self.train_set=name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)
        self.train_set=DataLoader(self.train_set,1,True,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
        print(f'train set len {len(self.train_set)}')
        self.val_set_list, self.val_set_names = [], []
        for val_set_cfg in self.cfg['val_set_list']:
            name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
            val_set = name2dataset[val_type](val_cfg, False)
            val_set = DataLoader(val_set,1,False,num_workers=self.cfg['worker_num'],collate_fn=dummy_collate_fn)
            self.val_set_list.append(val_set)
            self.val_set_names.append(name)
            print(f'{name} val set len {len(val_set)}')

    def _init_network(self):
        self.network=name2network[self.cfg['network']](self.cfg).cuda()
        if self.cfg['network'] == 'neuray_gen' and self.cfg['ft_load_path'] is not None:
            checkpoint=torch.load(self.cfg['ft_load_path'])
            self.network.load_state_dict(checkpoint['network_state_dict'])
            print('loading checkpoints for cross scene unmasked ft at path {}'.format(self.cfg['ft_load_path']))
        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu training for NeuRay
        if self.cfg['multi_gpus']:
            raise NotImplementedError
            # make multi gpu network
            # self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            # self.train_losses=[DummyLoss(self.val_losses)]
        else:
            self.train_network=self.network
            self.train_losses=self.val_losses

        if self.cfg['optimizer_type']=='adam':
            self.optimizer = Adam
        elif self.cfg['optimizer_type']=='sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator=ValidationEvaluator(self.cfg)
        self.lr_manager=name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])
        self.optimizer=self.lr_manager.construct_optimizer(self.optimizer,self.network)

    def __init__(self,cfg):
        self.cfg={**self.default_cfg,**cfg}
        self.model_name=cfg['name']
        self.model_dir=os.path.join(save_dir,'model', self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')

    def run(self, resume):
        self._init_dataset()
        self._init_network()
        self._init_logger()

        best_para,start_step=self._load_model(resume=resume)
        train_iter=iter(self.train_set)

        pbar=tqdm(total=self.cfg['total_step'],bar_format='{r_bar}')
        pbar.update(start_step)
        for step in range(start_step,self.cfg['total_step']+1):
            try:
                train_data = next(train_iter)
            except StopIteration:
                self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step']=step

            self.train_network.train()
            self.network.train()
            lr = self.lr_manager(self.optimizer, step)

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            log_info={}
            outputs=self.train_network(train_data)
            for loss in self.train_losses:
                loss_results = loss(outputs,train_data,step)
                for k,v in loss_results.items():
                    log_info[k]=v

            loss=0
            for k,v in log_info.items():
                if k.startswith('loss'):
                    loss=loss+torch.mean(v)

            loss.backward()
            self.optimizer.step()
            if (step % self.cfg['train_log_step']) == 0:
                self._log_data(log_info,step,'train')

            if (step % self.cfg['val_interval']) == 0 or (step == self.cfg['total_step']):
                torch.cuda.empty_cache()
                val_results={}
                val_para = 0
                for vi, val_set in enumerate(self.val_set_list):
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, val_set, step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k,v in val_results_cur.items():
                        val_results[f'{self.val_set_names[vi]}-{k}'] = v
                    # always use the last val set to select model!
                    val_para = val_para_cur

                if val_para>best_para:
                    print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para=val_para
                    self._save_model(step, best_para, save_fn=self.best_pth_fn)
                self._log_data(val_results,step,'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if (step %self.cfg['save_interval']) ==0:
                self._save_model(step ,best_para)

            if (step %self.cfg['extra_save_interval']) ==0:
                self._save_model(step ,best_para, suffix=True)
            
            if (self.cfg['stage'] == 'pretrain') and (self.cfg['mask_ratio'] > 0) and (step >= self.cfg['stage_two_begin']):
                self.network.updata_target()
            
            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),lr=lr)
            pbar.update(1)
            del loss, log_info

        pbar.close()

    def _load_model(self, resume):
        best_para,start_step=0,0
        load_pth = self.pth_fn
        if os.path.exists(load_pth) and resume:
            checkpoint=torch.load(load_pth)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('==> resuming ckpt from path--->', load_pth)
            print(f'==> resuming from step {start_step} best para {best_para}')
        else:
            print('-------training from scratch')
        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None, suffix=False):
        if save_fn is None:
            save_fn = self.pth_fn if not suffix else self.pth_fn.split('.pth')[0] + f'_{step}.pth'
        torch.save({
            'step':step,
            'best_para':best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },save_fn)

    def _init_logger(self):
        self.logger_dir = os.path.join(save_dir,'logs', self.model_name)
        os.makedirs(self.logger_dir, exist_ok=True)
        self.logger = Logger(self.logger_dir)

    def _log_data(self,results,step,prefix='train',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results,prefix,step,verbose)




