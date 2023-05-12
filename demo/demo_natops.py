# demo on NATOPS dataset
import sys
sys.path.append("/workspace/code/CVPR23_LFDM")

import argparse
import imageio
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
from PIL import Image
from misc import Logger, grid2fig, conf2fig
import random
from DM.modules.video_flow_diffusion_model import FlowDiffusion
from misc import resize
import cv2
import matplotlib.pyplot as plt

start = timeit.default_timer()
root_dir = '/data/hfn5052/text2motion/cvpr23/demo_natops'
GPU = "5"
postfix = "-j-of-lnc-upconv"
if "upconv" in postfix:
    use_deconv = False
    padding_mode = "reflect"
else:
    use_deconv = True
sampling_timesteps = 1000
ddim_sampling_eta = 1.0
use_residual_flow = "-rf" in postfix
learn_null_cond = "-lnc" in postfix
INPUT_SIZE = 128
N_FRAMES = 40
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
cond_scale = 1.
config_pth = "/workspace/code/CVPR23_LFDM/config/natops128.yaml"
# put your trained DM model here
RESTORE_FROM = "/data/hfn5052/text2motion/videoflowdiff_natops/snapshots-j-of-lnc-upconv/flowdiff_0020_S033600.pth"
# pu your trained LFAE model here
AE_RESTORE_FROM = "/data/hfn5052/text2motion/RegionMM/log-natops/natops128-crop/snapshots-crop/RegionMM_0100_S024000.pth"
CKPT_DIR = os.path.join(root_dir, "demo"+postfix)
os.makedirs(CKPT_DIR, exist_ok=True)
print(root_dir)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("cond scale:", cond_scale)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--fp16", default=False)
    return parser.parse_args()


args = get_arguments()


def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += np.array(MEAN)/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)

    model = FlowDiffusion(is_train=True,
                          sampling_timesteps=1000,
                          learn_null_cond=learn_null_cond,
                          use_deconv=use_deconv,
                          padding_mode=padding_mode,
                          ddim_sampling_eta=ddim_sampling_eta,
                          config_pth=config_pth,
                          pretrained_pth=AE_RESTORE_FROM)
    model.cuda()

    if args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            model.diffusion.load_state_dict(checkpoint['diffusion'])
            print("=> loaded checkpoint '{}'".format(args.restore_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
            exit(-1)
    else:
        print("NO checkpoint found!")
        exit(-1)

    model.eval()

    action_list = ["I Have Command",
                   "All Clear",
                   "Not Clear",
                   "Spread Wings",
                   "Fold Wings",
                   "Lock Wings",
                   "Up Hook",
                   "Down Hook",
                   "Remove Tiedowns",
                   "Remove Chocks",
                   "Insert Chocks",
                   "Move Ahead",
                   "Turn Left",
                   "Turn Right",
                   "Next Marshaller",
                   "Slow Down",
                   "Stop",
                   "Nosegear Steering",
                   "Hot Brakes",
                   "Brakes On",
                   "Brakes Off",
                   "Install Tiedowns",
                   "Fire",
                   "Cut Engine"]

    y_min, y_max, x_min, x_max = 10, 239, 30, 290

    ref_img_path = "/workspace/code/CVPR23_LFDM/demo/natops_examples/g01s10p01-0000-0055.png"
    ref_img_name = os.path.basename(ref_img_path)[:-4]
    ref_img_npy = imageio.v2.imread(ref_img_path)[:, :, :3]
    ref_img_npy = ref_img_npy[y_min:y_max, x_min:x_max, :]
    ref_img_npy = resize(ref_img_npy, 128, interpolation=cv2.INTER_AREA)
    ref_img_npy = np.asarray(ref_img_npy, np.float32)
    ref_img_npy = ref_img_npy - np.array(MEAN)
    ref_img = torch.from_numpy(ref_img_npy/255.0)
    ref_img = ref_img.permute(2, 0, 1).float()
    ref_imgs = ref_img.unsqueeze(dim=0).cuda()

    nf = 40
    cnt = 0
    for ref_text in action_list:
        model.set_sample_input(sample_img=ref_imgs, sample_text=[ref_text])
        model.sample_one_video(cond_scale=cond_scale)
        msk_size = ref_imgs.shape[-1]

        save_src_img = sample_img(ref_imgs, 0)
        new_im_list = []

        for frame_idx in range(nf):
            save_sample_out_img = sample_img(model.sample_out_vid[:, :, frame_idx], 0)
            save_sample_warp_img = sample_img(model.sample_warped_vid[:, :, frame_idx], 0)
            save_fake_grid = grid2fig(model.sample_vid_grid[0, :, frame_idx].permute((1, 2, 0)).data.cpu().numpy(),
                                      grid_size=32, img_size=msk_size)
            save_fake_conf = conf2fig(model.sample_vid_conf[0, :, frame_idx])
            new_im = Image.new('RGB', (msk_size * 5, msk_size))
            new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
            new_im.paste(Image.fromarray(save_sample_out_img, 'RGB'), (msk_size * 1, 0))
            new_im.paste(Image.fromarray(save_sample_warp_img, 'RGB'), (msk_size * 2, 0))
            new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, 0))
            new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, 0))
            new_im_arr = np.array(new_im)
            new_im_list.append(new_im_arr)

        video_name = "%04d_%s_%.2f.gif" % (cnt, ref_img_name, cond_scale)
        print(video_name)

        imageio.mimsave(os.path.join(CKPT_DIR, video_name), new_im_list)
        cnt += 1


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()

