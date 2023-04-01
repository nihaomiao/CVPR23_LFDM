import argparse

import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import timeit
import math
from PIL import Image
from misc import Logger, grid2fig, conf2fig
from datasets_natops import NATOPS
import sys
import random
from DM.modules.video_flow_diffusion_model import FlowDiffusion
from torch.optim.lr_scheduler import MultiStepLR

start = timeit.default_timer()
BATCH_SIZE = 20
MAX_EPOCH = 140
epoch_milestones = [80, 100]
root_dir = '/data/hfn5052/text2motion/videoflowdiff_natops'
data_dir = "/data/hfn5052/text2motion/dataset/NATOPS/split_img_data"
GPU = "0"
postfix = "-j-of-lnc-upconv"
joint = "joint" in postfix or "-j" in postfix  # allow joint training with unconditional model
if "random" in postfix:
    frame_sampling = "random"
elif "-vr" in postfix:
    frame_sampling = "very_random"
else:
    frame_sampling = "uniform"
only_use_flow = "onlyflow" in postfix or "-of" in postfix  # whether only use flow loss
if joint:
    null_cond_prob = 0.1
else:
    null_cond_prob = 0.0
if "upconv" in postfix:
    use_deconv = False
    padding_mode = "reflect"
else:
    use_deconv = True
use_residual_flow = "-rf" in postfix
learn_null_cond = "-lnc" in postfix
INPUT_SIZE = 128
N_FRAMES = 40
LEARNING_RATE = 2e-4
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
RESTORE_FROM = ""
AE_RESTORE_FROM = "/data/hfn5052/text2motion/RegionMM/log-natops/natops128-crop/snapshots-crop/RegionMM_0100_S024000.pth"
config_pth = "/workspace/code/CVPR23_LFDM/config/natops128.yaml"
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots'+postfix)
IMGSHOT_DIR = os.path.join(root_dir, 'imgshots'+postfix)
VIDSHOT_DIR = os.path.join(root_dir, "vidshots"+postfix)
SAMPLE_DIR = os.path.join(root_dir, 'sample'+postfix)
NUM_EXAMPLES_PER_EPOCH = 4800
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
SAVE_MODEL_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 4)
SAVE_VID_EVERY = 200
SAMPLE_VID_EVERY = 800
UPDATE_MODEL_EVERY = 400

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(IMGSHOT_DIR, exist_ok=True)
os.makedirs(VIDSHOT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(root_dir)
print("update saved model every:", UPDATE_MODEL_EVERY)
print("save model every:", SAVE_MODEL_EVERY)
print("save video every:", SAVE_VID_EVERY)
print("sample video every:", SAMPLE_VID_EVERY)
print(postfix)
print("RESTORE_FROM", RESTORE_FROM)
print("num examples per epoch:", NUM_EXAMPLES_PER_EPOCH)
print("max epoch:", MAX_EPOCH)
print("image size, num frames:", INPUT_SIZE, N_FRAMES)
print("epoch milestones:", epoch_milestones)
print("frame sampling:", frame_sampling)
print("only use flow loss:", only_use_flow)
print("null_cond_prob:", null_cond_prob)
print("use residual flow:", use_residual_flow)
print("learn null cond:", learn_null_cond)
print("use deconv:", use_deconv)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=2, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=20, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument('--save-vid-freq', default=SAVE_VID_EVERY, type=int)
    parser.add_argument('--sample-vid-freq', default=SAMPLE_VID_EVERY, type=int)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--n-frames", default=N_FRAMES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_MODEL_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--update-pred-every", type=int, default=UPDATE_MODEL_EVERY)
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--fp16", default=False)
    return parser.parse_args()


args = get_arguments()


def sample_img(rec_img_batch, idx=0):
    rec_img = rec_img_batch[idx].permute(1, 2, 0).data.cpu().numpy().copy()
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
                          img_size=INPUT_SIZE//4,
                          num_frames=N_FRAMES,
                          null_cond_prob=null_cond_prob,
                          sampling_timesteps=1000,
                          use_residual_flow=use_residual_flow,
                          learn_null_cond=learn_null_cond,
                          use_deconv=use_deconv,
                          padding_mode=padding_mode,
                          config_pth=config_pth,
                          pretrained_pth=AE_RESTORE_FROM)
    model.cuda()

    # Not set model to be train mode! Because pretrained flow autoenc need to be eval (BatchNorm)

    if args.fine_tune:
        pass
    elif args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            model_ckpt = model.diffusion.state_dict()
            for name, _ in model_ckpt.items():
                model_ckpt[name].copy_(checkpoint['diffusion'][name])
            model.diffusion.load_state_dict(model_ckpt)
            print("=> loaded checkpoint '{}'".format(args.restore_from))
            if args.set_start:
                if "optimizer_diff" in list(checkpoint.keys()):
                    model.optimizer_diff.load_state_dict(checkpoint['optimizer_diff'])
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
    else:
        print("NO checkpoint found!")

    setup_seed(args.random_seed)
    trainloader = data.DataLoader(NATOPS(data_dir=data_dir,
                                         image_size=INPUT_SIZE,
                                         num_frames=N_FRAMES,
                                         color_jitter=True,
                                         sampling=frame_sampling,
                                         mean=MEAN),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_warp = AverageMeter()

    cnt = 0
    actual_step = args.start_step
    start_epoch = int(math.ceil((args.start_step * args.batch_size)/NUM_EXAMPLES_PER_EPOCH))
    epoch_cnt = start_epoch

    scheduler = MultiStepLR(model.optimizer_diff, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)
    print("epoch %d, lr= %.7f" % (epoch_cnt, model.optimizer_diff.param_groups[0]["lr"]))

    while actual_step < args.final_step:
        iter_end = timeit.default_timer()

        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)

            real_vids, ref_texts, real_names = batch
            # use first frame of each video as reference frame
            ref_imgs = real_vids[:, :, 0, :, :].clone().detach()
            bs = real_vids.size(0)

            model.set_train_input(ref_img=ref_imgs, real_vid=real_vids, ref_text=ref_texts)
            model.optimize_parameters()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            losses.update(model.loss, bs)
            losses_rec.update(model.rec_loss, bs)
            losses_warp.update(model.rec_warp_loss, bs)

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.7f} ({loss.avg:.7f})\t'
                      'loss_rec {loss_rec.val:.4f} ({loss_rec.avg:.4f})\t'
                      'loss_warp {loss_warp.val:.4f} ({loss_warp.avg:.4f})'
                    .format(
                    cnt, actual_step, args.final_step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_rec=losses_rec,
                    loss_warp=losses_warp,
                ))

            null_cond_mask = np.array(model.diffusion.denoise_fn.null_cond_mask.data.cpu().numpy(),
                                      dtype=np.uint8)

            if actual_step % args.save_img_freq == 0:
                msk_size = ref_imgs.shape[-1]
                save_src_img = sample_img(ref_imgs)
                save_tar_img = sample_img(real_vids[:, :, N_FRAMES//2, :, :])
                save_real_out_img = sample_img(model.real_out_vid[:, :, N_FRAMES//2, :, :])
                save_real_warp_img = sample_img(model.real_warped_vid[:, :, N_FRAMES//2, :, :])
                save_fake_out_img = sample_img(model.fake_out_vid[:, :, N_FRAMES//2, :, :])
                save_fake_warp_img = sample_img(model.fake_warped_vid[:, :, N_FRAMES//2, :, :])
                save_real_grid = grid2fig(model.real_vid_grid[0, :, N_FRAMES//2].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=32, img_size=msk_size)
                save_fake_grid = grid2fig(model.fake_vid_grid[0, :, N_FRAMES//2].permute((1, 2, 0)).data.cpu().numpy(),
                                          grid_size=32, img_size=msk_size)
                save_real_conf = conf2fig(model.real_vid_conf[0, :, N_FRAMES//2])
                save_fake_conf = conf2fig(model.fake_vid_conf[0, :, N_FRAMES//2])
                new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size * 2, 0))
                new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                              + '_' + real_names[0] + "_%d.png" % (null_cond_mask[0])
                new_im_file = os.path.join(args.img_dir, new_im_name)
                new_im.save(new_im_file)

            if actual_step % args.save_vid_freq == 0:
                print("saving video...")
                num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(num_frames):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_real_out_img = sample_img(model.real_out_vid[:, :, nf, :, :])
                    save_real_warp_img = sample_img(model.real_warped_vid[:, :, nf, :, :])
                    save_fake_out_img = sample_img(model.fake_out_vid[:, :, nf, :, :])
                    save_fake_warp_img = sample_img(model.fake_warped_vid[:, :, nf, :, :])
                    save_real_grid = grid2fig(
                        model.real_vid_grid[0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_fake_grid = grid2fig(
                        model.fake_vid_grid[0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_real_conf = conf2fig(model.real_vid_conf[0, :, nf])
                    save_fake_conf = conf2fig(model.fake_vid_conf[0, :, nf])
                    new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                    new_im.paste(Image.fromarray(save_fake_out_img, 'RGB'), (msk_size * 2, 0))
                    new_im.paste(Image.fromarray(save_fake_warp_img, 'RGB'), (msk_size * 2, msk_size))
                    new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                    new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                    new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                    new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                               + '_' + real_names[0] + "_%d.gif" % (null_cond_mask[0])
                new_vid_file = os.path.join(VIDSHOT_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)

            # sampling
            if actual_step % args.sample_vid_freq == 0:
                print("sampling video...")
                model.set_sample_input(sample_img=ref_imgs[0].unsqueeze(dim=0),
                                       sample_text=[ref_texts[0]])
                model.sample_one_video(cond_scale=1.0)
                num_frames = real_vids.size(2)
                msk_size = ref_imgs.shape[-1]
                new_im_arr_list = []
                save_src_img = sample_img(ref_imgs)
                for nf in range(num_frames):
                    save_tar_img = sample_img(real_vids[:, :, nf, :, :])
                    save_real_out_img = sample_img(model.real_out_vid[:, :, nf, :, :])
                    save_real_warp_img = sample_img(model.real_warped_vid[:, :, nf, :, :])
                    save_sample_out_img = sample_img(model.sample_out_vid[:, :, nf, :, :])
                    save_sample_warp_img = sample_img(model.sample_warped_vid[:, :, nf, :, :])
                    save_real_grid = grid2fig(
                        model.real_vid_grid[0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_fake_grid = grid2fig(
                        model.sample_vid_grid[0, :, nf].permute((1, 2, 0)).data.cpu().numpy(),
                        grid_size=32, img_size=msk_size)
                    save_real_conf = conf2fig(model.real_vid_conf[0, :, nf])
                    save_fake_conf = conf2fig(model.fake_vid_conf[0, :, nf])
                    new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                    new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                    new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(save_real_warp_img, 'RGB'), (msk_size, msk_size))
                    new_im.paste(Image.fromarray(save_sample_out_img, 'RGB'), (msk_size * 2, 0))
                    new_im.paste(Image.fromarray(save_sample_warp_img, 'RGB'), (msk_size * 2, msk_size))
                    new_im.paste(Image.fromarray(save_real_grid, 'RGB'), (msk_size * 3, 0))
                    new_im.paste(Image.fromarray(save_fake_grid, 'RGB'), (msk_size * 3, msk_size))
                    new_im.paste(Image.fromarray(save_real_conf, 'L'), (msk_size * 4, 0))
                    new_im.paste(Image.fromarray(save_fake_conf, 'L'), (msk_size * 4, msk_size))
                    new_im_arr = np.array(new_im)
                    new_im_arr_list.append(new_im_arr)
                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                               + '_' + real_names[0] + ".gif"
                new_vid_file = os.path.join(SAMPLE_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)

            # save model at i-th step
            if actual_step % args.save_pred_every == 0:
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'diffusion': model.module.diffusion.state_dict(),
                            'optimizer_diff': model.optimizer_diff.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            # update saved model
            if actual_step % args.update_pred_every == 0:
                print('updating saved snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'diffusion': model.module.diffusion.state_dict(),
                            'optimizer_diff': model.optimizer_diff.state_dict()},
                           osp.join(args.snapshot_dir, 'flowdiff.pth'))

            if actual_step >= args.final_step:
                break

            cnt += 1

        scheduler.step()
        epoch_cnt += 1
        print("epoch %d, lr= %.7f" % (epoch_cnt, model.optimizer_diff.param_groups[0]["lr"]))

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'diffusion': model.module.diffusion.state_dict(),
                'optimizer_diff': model.optimizer_diff.state_dict()},
               osp.join(args.snapshot_dir,
                        'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))
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

