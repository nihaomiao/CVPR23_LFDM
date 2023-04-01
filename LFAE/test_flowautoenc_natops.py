import sys
import argparse

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import timeit
from PIL import Image
from misc import grid2fig
from DM.datasets_natops import NATOPS_test

import random
from LFAE.modules.flow_autoenc import FlowAE
import torch.nn.functional as F
import matplotlib.pyplot as plt
from LFAE.modules.util import Visualizer
import json_tricks as json


start = timeit.default_timer()
BATCH_SIZE = 10
root_dir = '/data/hfn5052/text2motion/flowautoenc_video_natops'
data_dir = "/data/hfn5052/text2motion/dataset/NATOPS/split_img_data"
GPU = "6"
postfix = "-crop"
INPUT_SIZE = 128
N_FRAMES = 40
NUM_VIDEOS = 1000
SAVE_VIDEO = False
NUM_ITER = NUM_VIDEOS // BATCH_SIZE
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
# put your trained LFAE model here
RESTORE_FROM = "/data/hfn5052/text2motion/RegionMM/log-natops/natops128-crop/snapshots-crop/RegionMM_0100_S024000.pth"
config_pth = "/workspace/code/CVPR23_LFDM/config/natops128.yaml"
CKPT_DIR = os.path.join(root_dir, "flowae-res"+postfix)
os.makedirs(CKPT_DIR, exist_ok=True)
json_path = os.path.join(CKPT_DIR, "loss%d%s.json" % (NUM_VIDEOS, postfix))
visualizer = Visualizer()
print(root_dir)
print(postfix)
print("RESTORE_FROM:", RESTORE_FROM)
print(json_path)
print(config_pth)
print("save video:", SAVE_VIDEO)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
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

    model = FlowAE(is_train=False, config_pth=config_pth)
    model.cuda()

    if os.path.isfile(args.restore_from):
        print("=> loading checkpoint '{}'".format(args.restore_from))
        checkpoint = torch.load(args.restore_from)
        model.generator.load_state_dict(checkpoint['generator'])
        model.region_predictor.load_state_dict(checkpoint['region_predictor'])
        model.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
        print("=> loaded checkpoint '{}'".format(args.restore_from))
    else:
        print("=> no checkpoint found at '{}'".format(args.restore_from))
        exit(-1)

    model.eval()

    setup_seed(args.random_seed)

    testloader = data.DataLoader(NATOPS_test(data_dir=data_dir,
                                             image_size=args.input_size,
                                             num_frames=N_FRAMES,
                                             mean=MEAN),
                                 batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    iter_end = timeit.default_timer()
    cnt = 0

    out_loss = 0.0
    warp_loss = 0.0
    num_sample = 0.0
    l1_loss = torch.nn.L1Loss(reduction='sum')
    global_iter = 0

    while global_iter < NUM_ITER:
        for i_iter, batch in enumerate(testloader):
            if i_iter >= NUM_ITER:
                break
            if global_iter >= NUM_ITER:
                break

            data_time.update(timeit.default_timer() - iter_end)

            real_vids, ref_texts, real_names = batch
            # use first frame of each video as reference frame
            ref_imgs = real_vids[:, :, 0, :, :].clone().detach()
            bs = real_vids.size(0)

            batch_time.update(timeit.default_timer() - iter_end)

            nf = real_vids.size(2)
            out_img_list = []
            warped_img_list = []
            warped_grid_list = []
            conf_map_list = []
            for frame_idx in range(nf):
                dri_imgs = real_vids[:, :, frame_idx, :, :]
                with torch.no_grad():
                    model.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
                    model.forward()
                out_img_list.append(model.generated['prediction'].clone().detach())
                warped_img_list.append(model.generated['deformed'].clone().detach())
                warped_grid_list.append(model.generated['optical_flow'].clone().detach())
                conf_map_list.append(model.generated['occlusion_map'].clone().detach())

            out_img_list_tensor = torch.stack(out_img_list, dim=0)
            warped_img_list_tensor = torch.stack(warped_img_list, dim=0)
            warped_grid_list_tensor = torch.stack(warped_grid_list, dim=0)
            conf_map_list_tensor = torch.stack(conf_map_list, dim=0)

            out_loss += l1_loss(real_vids.permute(2, 0, 1, 3, 4).cpu(), out_img_list_tensor.cpu()).item()
            warp_loss += l1_loss(real_vids.permute(2, 0, 1, 3, 4).cpu(), warped_img_list_tensor.cpu()).item()
            num_sample += bs

            if SAVE_VIDEO:
                for batch_idx in range(bs):
                    msk_size = ref_imgs.shape[-1]
                    new_im_list = []
                    for frame_idx in range(nf):
                        save_tar_img = sample_img(real_vids[:, :, frame_idx], batch_idx)
                        save_out_img = sample_img(out_img_list_tensor[frame_idx], batch_idx)
                        save_warped_img = sample_img(warped_img_list_tensor[frame_idx], batch_idx)
                        save_warped_grid = grid2fig(warped_grid_list_tensor[frame_idx, batch_idx].data.cpu().numpy(),
                                                    grid_size=32, img_size=msk_size)
                        save_conf_map = conf_map_list_tensor[frame_idx, batch_idx].unsqueeze(dim=0)
                        save_conf_map = save_conf_map.data.cpu()
                        save_conf_map = F.interpolate(save_conf_map, size=real_vids.shape[3:5]).numpy()
                        save_conf_map = np.transpose(save_conf_map, [0, 2, 3, 1])
                        save_conf_map = np.array(save_conf_map[0, :, :, 0]*255, dtype=np.uint8)
                        new_im = Image.new('RGB', (msk_size * 5, msk_size))
                        new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
                        new_im.paste(Image.fromarray(save_out_img, 'RGB'), (msk_size, 0))
                        new_im.paste(Image.fromarray(save_warped_img, 'RGB'), (msk_size * 2, 0))
                        new_im.paste(Image.fromarray(save_warped_grid), (msk_size * 3, 0))
                        new_im.paste(Image.fromarray(save_conf_map, "L"), (msk_size * 4, 0))
                        new_im_list.append(new_im)
                    video_name = "%04d_%s.gif" % (cnt, real_names[batch_idx])
                    imageio.mimsave(os.path.join(CKPT_DIR, video_name), new_im_list)
                    cnt += 1

            iter_end = timeit.default_timer()

            if global_iter % args.print_freq == 0:
                print('Test:[{0}/{1}]\t'
                      'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                      .format(global_iter, NUM_ITER, batch_time=batch_time))
            global_iter += 1

    print("loss for prediction: %.5f" % (out_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)))
    print("loss for warping: %.5f" % (warp_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)))

    res_dict = {}
    res_dict["out_loss"] = out_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)
    res_dict["warp_loss"] = warp_loss/(num_sample*INPUT_SIZE*INPUT_SIZE*3)
    with open(json_path, "w") as f:
        json.dump(res_dict, f)

    end = timeit.default_timer()
    print(end - start, 'seconds')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()

