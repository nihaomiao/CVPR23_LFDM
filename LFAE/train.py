# train a LFAE
# this code is based on RegionMM (MRAA): https://github.com/snap-research/articulated-animation
import os.path
import torch
from torch.utils.data import DataLoader
from modules.model import ReconstructionModel
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import DatasetRepeater
import timeit
from modules.util import Visualizer
import imageio
import math


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


def train(config, generator, region_predictor, bg_predictor, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer = torch.optim.Adam(list(generator.parameters()) +
                                 list(region_predictor.parameters()) +
                                 list(bg_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))

    start_epoch = 0
    start_step = 0
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        if config["set_start"]:
            start_step = int(math.ceil(ckpt['example'] / config['train_params']['batch_size']))
            start_epoch = ckpt['epoch']
        generator.load_state_dict(ckpt['generator'])
        region_predictor.load_state_dict(ckpt['region_predictor'])
        bg_predictor.load_state_dict(ckpt['bg_predictor'])
        if 'optimizer' in list(ckpt.keys()):
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except:
                optimizer.load_state_dict(ckpt['optimizer'].state_dict())

    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    model = ReconstructionModel(region_predictor, bg_predictor, generator, train_params)

    visualizer = Visualizer(**config['visualizer_params'])

    if torch.cuda.is_available():
        if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
            model = DataParallelWithCallback(model, device_ids=device_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    # rewritten by nhm
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_losses = AverageMeter()
    losses_perc = AverageMeter()
    losses_equiv_shift = AverageMeter()
    losses_equiv_affine = AverageMeter()

    cnt = 0
    epoch_cnt = start_epoch
    actual_step = start_step
    final_step = config["num_step_per_epoch"] * train_params["max_epochs"]

    while actual_step < final_step:
        iter_end = timeit.default_timer()

        for i_iter, x in enumerate(dataloader):
            actual_step = int(start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)
            optimizer.zero_grad()
            losses, generated = model(x)
            loss_values = [val.mean() for val in losses.values()]
            loss = sum(loss_values)
            loss.backward()
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            bs = x['source'].size(0)
            total_losses.update(loss, bs)
            losses_perc.update(loss_values[0], bs)
            losses_equiv_shift.update(loss_values[1], bs)
            losses_equiv_affine.update(loss_values[2], bs)

            if actual_step % train_params["print_freq"] == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'loss_perc {loss_perc.val:.4f} ({loss_perc.avg:.4f})\n'
                      'loss_shift {loss_shift.val:.4f} ({loss_shift.avg:.4f})\t'
                      'loss_affine {loss_affine.val:.4f} ({loss_affine.avg:.4f})'
                    .format(
                    cnt, actual_step, final_step,
                    loss=total_losses,
                    loss_perc=losses_perc,
                    loss_shift=losses_equiv_shift,
                    loss_affine=losses_equiv_affine
                ))

            if actual_step % train_params['save_img_freq'] == 0:
                save_image = visualizer.visualize(x['driving'], x['source'], generated, index=0)
                save_name = 'B' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") \
                            + '_' + x["frame"][0][0][:-4] + '_to_' + x["frame"][1][0][-7:]
                save_file = os.path.join(config["imgshots"], save_name)
                imageio.imsave(save_file, save_image)

            if actual_step % config["save_ckpt_freq"] == 0 and cnt != 0:
                print('taking snapshot...')
                torch.save({'example': actual_step * train_params["batch_size"],
                            'epoch': epoch_cnt,
                            'generator': generator.state_dict(),
                            'bg_predictor': bg_predictor.state_dict(),
                            'region_predictor': region_predictor.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(config["snapshots"],
                                        'RegionMM_' + format(train_params["batch_size"], "04d") +
                                        '_S' + format(actual_step, "06d") + '.pth'))

            if actual_step % train_params["update_ckpt_freq"] == 0 and cnt != 0:
                print('updating snapshot...')
                torch.save({'example': actual_step * train_params["batch_size"],
                            'epoch': epoch_cnt,
                            'generator': generator.state_dict(),
                            'bg_predictor': bg_predictor.state_dict(),
                            'region_predictor': region_predictor.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(config["snapshots"], 'RegionMM.pth'))

            if actual_step >= final_step:
                break

            cnt += 1

        scheduler.step()
        epoch_cnt += 1
        # print lr
        print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))

    print('save the final model...')
    torch.save({'example': actual_step * train_params["batch_size"],
                'epoch': epoch_cnt,
                'generator': generator.state_dict(),
                'bg_predictor': bg_predictor.state_dict(),
                'region_predictor': region_predictor.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(config["snapshots"],
                            'RegionMM_' + format(train_params["batch_size"], "04d") +
                            '_S' + format(actual_step, "06d") + '.pth'))


