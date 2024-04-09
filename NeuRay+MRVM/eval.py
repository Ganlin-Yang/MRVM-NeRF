import argparse
import os
import lpips

import torch
from skimage.io import imread
from tqdm import tqdm
import numpy as np

from utils.base_utils import color_map_forward
from skimage.measure import compare_ssim, compare_psnr
class Evaluator:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='vgg').cuda().eval()
        # self.loss_fn_alex = lpips.LPIPS(net='alex').cuda().eval()

    def eval_metrics_img(self,gt_img, pr_img):
        psnr = compare_psnr(pr_img, gt_img, data_range = 1)
        ssim = compare_ssim(pr_img, gt_img, multichannel=True, data_range = 1)
        with torch.no_grad():
            gt_img_th = torch.from_numpy(gt_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            pr_img_th = torch.from_numpy(pr_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            score = float(self.loss_fn_alex(gt_img_th, pr_img_th).flatten()[0].cpu().numpy())
        return float(psnr), float(ssim), score


    def eval(self, dir_gt, dir_pr):
        results=[]
        num = len(os.listdir(dir_gt))
        for k in tqdm(range(0, num)):
            pr_img = imread(f'{dir_pr}/{k}-nr_fine.jpg')
            gt_img = imread(f'{dir_gt}/{k}.jpg')

            psnr, ssim, lpips_score = self.eval_metrics_img(gt_img, pr_img)
            results.append([psnr,ssim,lpips_score])
        psnr, ssim, lpips_score = np.mean(np.asarray(results),0)

        msg=f'psnr {psnr:.4f} ssim {ssim:.4f} lpips {lpips_score:.4f}'
        print(msg)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_gt', type=str, default='/home/v-yangganlin/NeuRay/data/render/llff_colmap/fern/high/gt')
    parser.add_argument('--dir_pr', type=str, default='/home/v-yangganlin/NeuRay/data/render/llff_colmap/fern/high/neuray_gen_depth-pretrain-eval')
    flags = parser.parse_args()
    evaluator = Evaluator()
    evaluator.eval(flags.dir_gt, flags.dir_pr)
