import sys
import os


import argparse
from pyhocon import ConfigFactory


def parse_args(
    callback=None,
    training=False,
    default_conf="conf/default_mv.conf",
    default_expname="example",
    default_data_format="dvr",
    default_num_epochs=10000000,
    default_lr=1e-4,
    default_gamma=1.00,
    default_datadir="data",
    default_ray_batch_size=50000,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", type=str, default=None)
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited"
    )
    parser.add_argument(
        "--name", "-n", type=str, default='sn64', help="experiment name"
    )
    parser.add_argument(
        "--dataset_format",
        "-F",
        type=str,
        default=None,
        help="Dataset format, multi_obj | dvr | dvr_gen | dvr_dtu | srn",
    )
    parser.add_argument(
        "--exp_group_name",
        "-G",
        type=str,
        default=None,
        help="if we want to group some experiments together",
    )
    parser.add_argument(
        "--logs_path", type=str, default="logs", help="logs output directory",
    )
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default="checkpoints",
        help="checkpoints output directory",
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="visuals",
        help="visualization output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000000000,
        help="number of epochs to train for, in practice we don not use this argument",
    )
    parser.add_argument('--itera', type=int, default=100000)
    parser.add_argument("--lr", type=float, default=default_lr, help="learning rate")
    parser.add_argument(
        "--gamma", type=float, default=default_gamma, help="learning rate decay factor"
    )
    parser.add_argument(
        "--warm_up_iter", type=int, default=5000, help="warm up iters"
    )

    parser.add_argument(
        "--datadir", "-D", type=str, default=None, help="Dataset directory"
    )
    parser.add_argument(
        "--ray_batch_size", "-R", type=int, default=default_ray_batch_size, help="Ray batch size"
    )
    parser.add_argument(
        "--num_worker", type=int, default=8,
    )
    parser.add_argument('--stage', type=str, default='pretrain', help='pretrain or fine_tune')
    parser.add_argument('--mask_ratio', type = float, default = 0.5)
    parser.add_argument('--lambda_coarse', type = float, default = 1.0)
    parser.add_argument('--lambda_fine', type = float, default = 1.0)
    parser.add_argument('--lambda_recons', type = float, default = 1.0)
    parser.add_argument('--stage_two_begin', type=int, default=1)
    parser.add_argument('--stage_two_warmup', type=int, default=4)
    parser.add_argument('--EMA', type=float, default=0.99)
    parser.add_argument('--load_iter', type=int, default=600000, help='choose from 120000, 160000, 200000')
    parser.add_argument('--path_to_load', default=None, type=str)
    if callback is not None:
        parser = callback(parser)
    args = parser.parse_args()

    if args.exp_group_name is not None:
        args.logs_path = os.path.join(args.logs_path, args.exp_group_name)
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.exp_group_name)
        args.visual_path = os.path.join(args.visual_path, args.exp_group_name)

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    EXPCONF_PATH = os.path.join(PROJECT_ROOT, "expconf.conf")
    expconf = ConfigFactory.parse_file(EXPCONF_PATH)

    if args.conf is None:
        args.conf = expconf.get_string("config." + args.name, default_conf)

    if args.datadir is None:
        args.datadir = expconf.get_string("datadir." + args.name, default_datadir)

    conf = ConfigFactory.parse_file(args.conf)

    if args.dataset_format is None:
        args.dataset_format = conf.get_string("data.format", default_data_format)

    args.gpu_id = list(map(int, args.gpu_id.split(',')))

    print("EXPERIMENT NAME:", args.name)
    if training:
        print("CONTINUE?", "yes" if args.resume else "no")
    print("* Config file:", args.conf)
    print("* Dataset format:", args.dataset_format)
    print("* Dataset location:", args.datadir)
    return args, conf
