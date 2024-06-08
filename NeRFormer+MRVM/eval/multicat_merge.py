"""
Compute metrics on rendered images (after eval.py).
First computes per-object metric then reduces them. If --multicat is used
then also summarized per-categority metrics. Use --reduce_only to skip the
per-object computation step.

Note eval.py already outputs PSNR/SSIM.
This also computes LPIPS and is useful for double-checking metric is correct.
"""

import os
import os.path as osp
import argparse
import skimage.measure
from tqdm import tqdm
import warnings
import lpips
import numpy as np
import torch
import imageio
import json

parser = argparse.ArgumentParser(description="Calculate PSNR for rendered images.")

parser.add_argument(
    "--output",
    "-O",
    type=str,
    default="./output_list/sn64.txt",
    help="Root path of rendered output (our format, from eval.py)",
)
parser.add_argument(
    "--multicat",
    action="store_true",
    help="Prepend category id to object id. Specify if model fits multiple categories.",
)
parser.add_argument(
    "--metadata",
    type=str,
    default="/data/nerformer_data/NMR_Dataset/metadata.yaml",
    help="Path to dataset metadata under datadir, used for getting category names if --multicat",
)
parser.add_argument(
    "--finish_txt_path",
    '-f',
    default='./input_list/sn64.txt',
)
args = parser.parse_args()

out_metrics_path = args.output
input_txt = args.finish_txt_path

def run_reduce():
    if args.multicat:
        meta = json.load(open(args.metadata, "r"))
        cats = sorted(list(meta.keys()))
        cat_description = {cat: meta[cat]["name"].split(",")[0] for cat in cats}

    with open(input_txt, "r") as f:
        metrics = [line.split() for line in f.readlines()]
    
    all_objs = [metrics[i][0] for i in range(len(metrics))]
    psnrs = [metrics[i][1] for i in range(len(metrics))]
    ssims = [metrics[i][2] for i in range(len(metrics))]
    lpipss = [metrics[i][3] for i in range(len(metrics))]
    
    print(">>> PROCESSING", len(all_objs), "OBJECTS")

    METRIC_NAMES = ["psnr", "ssim", "lpips"]

    if args.multicat:
        cat_sz = {}
        for cat in cats:
            cat_sz[cat] = 0

    all_metrics = {}
    for name in METRIC_NAMES:
        if args.multicat:
            for cat in cats:
                all_metrics[cat + "." + name] = 0.0
        all_metrics[name] = 0.0

    for obj_id in range(len(all_objs)):
        if args.multicat:
            cat_name = all_objs[obj_id].split("_")[0]
            cat_sz[cat_name] += 1
            all_metrics[cat_name + "." + "psnr"] += float(psnrs[obj_id])
            all_metrics[cat_name + "." + "ssim"] += float(ssims[obj_id])
            all_metrics[cat_name + "." + "lpips"] += float(lpipss[obj_id])
        all_metrics["psnr"] += float(psnrs[obj_id])
        all_metrics["ssim"] += float(ssims[obj_id])
        all_metrics["lpips"] += float(lpipss[obj_id])

    for name in METRIC_NAMES:
        if args.multicat:
            for cat in cats:
                if cat_sz[cat] > 0:
                    all_metrics[cat + "." + name] /= cat_sz[cat]
        all_metrics[name] /= len(all_objs)

    metrics_txt = []
    if args.multicat:
        for cat in cats:
            if cat_sz[cat] > 0:
                cat_txt = "{:12s}".format(cat_description[cat])
                for name in METRIC_NAMES:
                    cat_txt += " {}: {:.6f}".format(name, all_metrics[cat + "." + name])
                cat_txt += " n_inst: {}".format(cat_sz[cat])
                metrics_txt.append(cat_txt)

        total_txt = "---\n{:12s}".format("total")
    else:
        total_txt = ""
    for name in METRIC_NAMES:
        total_txt += " {}: {:.6f}".format(name, all_metrics[name])
    metrics_txt.append(total_txt)

    metrics_txt = "\n".join(metrics_txt)
    with open(out_metrics_path, "w") as f:
        f.write(metrics_txt)
    print("WROTE", out_metrics_path)
    


if __name__ == "__main__":
    print(">>> Reduce")
    run_reduce()
