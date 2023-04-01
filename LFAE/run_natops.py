# Estimate flow and occlusion mask via RegionMM for NATOPS dataset

import os
import sys
import math
import yaml
from argparse import ArgumentParser
from shutil import copy

from natops_dataset import FramesDataset

from modules.generator import Generator
from modules.bg_motion_predictor import BGMotionPredictor
from modules.region_predictor import RegionPredictor
from modules.avd_network import AVDNetwork

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

from train import train


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    cudnn.enabled = True
    cudnn.benchmark = True

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--postfix", default="-crop")
    parser.add_argument("--random-seed", default=1234)
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--config",
                        default="/workspace/code/CVPR23_LFDM/config/natops128.yaml",
                        help="path to config")
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--log_dir",
                        default='/data/hfn5052/text2motion/RegionMM/log-natops',
                        help="path to log into")
    parser.add_argument("--checkpoint",  # use the pretrained Taichi model provided by Snap
                        default="/data/hfn5052/text2motion/RegionMM/taichi256.pth",
                        help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", default=False, help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()

    setup_seed(opt.random_seed)

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0]+opt.postfix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    config["snapshots"] = os.path.join(log_dir, 'snapshots'+opt.postfix)
    os.makedirs(config["snapshots"], exist_ok=True)
    config["imgshots"] = os.path.join(log_dir, 'imgshots'+opt.postfix)
    os.makedirs(config["imgshots"], exist_ok=True)
    config["set_start"] = opt.set_start
    log_txt = os.path.join(log_dir,
                           "B"+format(config['train_params']['batch_size'], "04d")+
                           "E"+format(config['train_params']['max_epochs'], "04d")+".log")
    sys.stdout = Logger(log_txt, sys.stdout)

    print("postfix:", opt.postfix)
    print("checkpoint:", opt.checkpoint)
    print("batch size:", config['train_params']['batch_size'])

    generator = Generator(num_regions=config['model_params']['num_regions'],
                          num_channels=config['model_params']['num_channels'],
                          revert_axis_swap=config['model_params']['revert_axis_swap'],
                          **config['model_params']['generator_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                       num_channels=config['model_params']['num_channels'],
                                       estimate_affine=config['model_params']['estimate_affine'],
                                       **config['model_params']['region_predictor_params'])

    if torch.cuda.is_available():
        region_predictor.to(opt.device_ids[0])

    if opt.verbose:
        print(region_predictor)

    bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                     **config['model_params']['bg_predictor_params'])
    if torch.cuda.is_available():
        bg_predictor.to(opt.device_ids[0])
    if opt.verbose:
        print(bg_predictor)

    avd_network = AVDNetwork(num_regions=config['model_params']['num_regions'],
                             **config['model_params']['avd_network_params'])
    if torch.cuda.is_available():
        avd_network.to(opt.device_ids[0])
    if opt.verbose:
        print(avd_network)

    config['dataset_params']['use_crop'] = "crop" in opt.postfix
    dataset = FramesDataset(**config['dataset_params'])
    config["num_example_per_epoch"] = config['train_params']['num_repeats'] * len(dataset)
    config["num_step_per_epoch"] = math.ceil(config["num_example_per_epoch"]/float(config['train_params']['batch_size']))
    config["save_ckpt_freq"] = config["num_step_per_epoch"] * (config['train_params']['max_epochs'] // 5)
    print("save ckpt freq:", config["save_ckpt_freq"])

    print("Training...")
    train(config, generator, region_predictor, bg_predictor, opt.checkpoint, log_dir, dataset, opt.device_ids)

