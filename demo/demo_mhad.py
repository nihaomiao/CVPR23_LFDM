import sys
sys.path.append("/workspace/code/CVPR23_LFDM")  # change this to your code directory

import argparse
import imageio
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
from PIL import Image
from misc import grid2fig, conf2fig
import random
from DM.modules.video_flow_diffusion_model import FlowDiffusion
from misc import resize
import cv2
import matplotlib.pyplot as plt

start = timeit.default_timer()
root_dir = '/data/hfn5052/text2motion/cvpr23/demo_mhad'
GPU = "1"
postfix = "-j-sl-random-of-tr-rmm"
INPUT_SIZE = 128
N_FRAMES = 40
RANDOM_SEED = 2222
MEAN = (0.0, 0.0, 0.0)
cond_scale = 1.
# downloaded the pretrained DM model and put its path here
RESTORE_FROM = "/data/hfn5052/text2motion/videoflowdiff/snapshots-joint-steplr-random-onlyflow-train-regionmm/" \
               "flowdiff_0006_S086400.pth"
# downloaded the pretrained LFAE model and put its path here
AE_RESTORE_FROM = "/data/hfn5052/text2motion/RegionMM/log/mhad128/snapshots/RegionMM_0100_S043100.pth"
config_pth = "/workspace/code/CVPR23_LFDM/config/mug128.yaml"
CKPT_DIR = os.path.join(root_dir, "demo"+postfix)
os.makedirs(CKPT_DIR, exist_ok=True)
# IMG_DIR = os.path.join(root_dir, "demo_img"+postfix)
# os.makedirs(IMG_DIR, exist_ok=True)
print(root_dir)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print("AE_RESTORE_FROM:", AE_RESTORE_FROM)
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
                          pretrained_pth=AE_RESTORE_FROM,
                          config_pth=config_pth)
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

    action_list = ["right arm swipe to the left",
                   "right arm swipe to the right",
                   "right hand wave",
                   "two hand front clap",
                   "right arm throw",
                   "cross arms in the chest",
                   "basketball shooting",
                   "draw x",
                   "draw circle clockwise",
                   "draw circle counter clockwise",
                   "draw triangle",
                   "right hand bowling",
                   "front boxing",
                   "baseball swing from right",
                   "tennis forehand swing",
                   "two arms curl",
                   "tennis serve",
                   "two hand push",
                   "knock on door",
                   "hand catch",
                   "pick up and throw",
                   "jogging",
                   "walking",
                   "stand to sit",
                   "forward lunge (left foot forward)",
                   "squat"]

    ref_img_path = "/workspace/code/CVPR23_LFDM/demo/mhad_examples/a11_s4_t1_000.png"
    ref_img_name = os.path.basename(ref_img_path)[:-4]
    ref_img_npy = imageio.v2.imread(ref_img_path)[:, :, :3]
    ref_img_npy = cv2.resize(ref_img_npy, (336, 480), interpolation=cv2.INTER_AREA)
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

        # img_dir_name = "%04d_%s_%.2f" % (cnt, ref_img_name, cond_scale)
        # cur_img_dir = os.path.join(IMG_DIR, img_dir_name)
        # os.makedirs(cur_img_dir, exist_ok=True)

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
            # # save frame
            # new_im_name = "%03d_%04d_%s_%.2f.png" % (frame_idx, cnt, ref_img_name, cond_scale)
            # imageio.imsave(os.path.join(cur_img_dir, new_im_name), new_im_arr)

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

