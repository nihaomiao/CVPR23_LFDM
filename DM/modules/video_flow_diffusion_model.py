# use diffusion model to generate pseudo ground truth flow volume based on RegionMM
# 3D noise to 3D flow
# flow size: 2*32*32*40
# some codes based on https://github.com/lucidrains/video-diffusion-pytorch

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from LFAE.modules.generator import Generator
from LFAE.modules.bg_motion_predictor import BGMotionPredictor
from LFAE.modules.region_predictor import RegionPredictor
from DM.modules.video_flow_diffusion import Unet3D, GaussianDiffusion
import yaml


class FlowDiffusion(nn.Module):
    def __init__(self, img_size=32, num_frames=40, sampling_timesteps=250,
                 null_cond_prob=0.1, ddim_sampling_eta=1., timesteps=1000,
                 dim_mults=(1, 2, 4, 8),
                 lr=1e-4, adam_betas=(0.9, 0.99), is_train=True,
                 only_use_flow=True,
                 use_residual_flow=False,
                 learn_null_cond=False,
                 use_deconv=True,
                 padding_mode="zeros",
                 pretrained_pth="",
                 config_pth=""):
        super(FlowDiffusion, self).__init__()
        self.use_residual_flow = use_residual_flow
        self.only_use_flow = only_use_flow

        if pretrained_pth != "":
            checkpoint = torch.load(pretrained_pth)
        with open(config_pth) as f:
            config = yaml.safe_load(f)

        self.generator = Generator(num_regions=config['model_params']['num_regions'],
                                   num_channels=config['model_params']['num_channels'],
                                   revert_axis_swap=config['model_params']['revert_axis_swap'],
                                   **config['model_params']['generator_params']).cuda()
        if pretrained_pth != "":
            self.generator.load_state_dict(checkpoint['generator'])
            self.generator.eval()
            self.set_requires_grad(self.generator, False)

        self.region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                                num_channels=config['model_params']['num_channels'],
                                                estimate_affine=config['model_params']['estimate_affine'],
                                                **config['model_params']['region_predictor_params']).cuda()
        if pretrained_pth != "":
            self.region_predictor.load_state_dict(checkpoint['region_predictor'])
            self.region_predictor.eval()
            self.set_requires_grad(self.region_predictor, False)

        self.bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                              **config['model_params']['bg_predictor_params'])
        if pretrained_pth != "":
            self.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
            self.bg_predictor.eval()
            self.set_requires_grad(self.bg_predictor, False)

        self.unet = Unet3D(dim=64,
                           channels=3 + 256,
                           out_grid_dim=2,
                           out_conf_dim=1,
                           dim_mults=dim_mults,
                           use_bert_text_cond=True,
                           learn_null_cond=learn_null_cond,
                           use_final_activation=False,
                           use_deconv=use_deconv,
                           padding_mode=padding_mode)

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=img_size,
            num_frames=num_frames,
            sampling_timesteps=sampling_timesteps,
            timesteps=timesteps,  # number of steps
            loss_type='l2',  # L1 or L2
            use_dynamic_thres=True,
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta,
        )

        self.ref_img = None
        self.ref_img_fea = None
        self.real_vid = None
        self.real_out_vid = None
        self.real_warped_vid = None
        self.real_vid_grid = None
        self.real_vid_conf = None

        self.fake_out_vid = None
        self.fake_warped_vid = None
        self.fake_vid_grid = None
        self.fake_vid_conf = None

        self.sample_out_vid = None
        self.sample_warped_vid = None
        self.sample_vid_grid = None
        self.sample_vid_conf = None

        # training
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()
            self.lr = lr
            self.loss = torch.tensor(0.0).cuda()
            self.rec_loss = torch.tensor(0.0).cuda()
            self.rec_warp_loss = torch.tensor(0.0).cuda()
            self.optimizer_diff = torch.optim.Adam(self.diffusion.parameters(),
                                                   lr=lr, betas=adam_betas)

    def forward(self):
        # compute pseudo ground-truth flow
        b, _, nf, H, W = self.real_vid.size()

        real_grid_list = []
        real_conf_list = []
        real_out_img_list = []
        real_warped_img_list = []
        with torch.no_grad():
            source_region_params = self.region_predictor(self.ref_img)
            for idx in range(nf):
                driving_region_params = self.region_predictor(self.real_vid[:, :, idx, :, :])
                bg_params = self.bg_predictor(self.ref_img, self.real_vid[:, :, idx, :, :])
                generated = self.generator(self.ref_img, source_region_params=source_region_params,
                                           driving_region_params=driving_region_params, bg_params=bg_params)
                generated.update({'source_region_params': source_region_params,
                                  'driving_region_params': driving_region_params})
                real_grid_list.append(generated["optical_flow"].permute(0, 3, 1, 2))
                # normalized occlusion map
                real_conf_list.append(generated["occlusion_map"])
                real_out_img_list.append(generated["prediction"])
                real_warped_img_list.append(generated["deformed"])
        self.real_vid_grid = torch.stack(real_grid_list, dim=2)
        self.real_vid_conf = torch.stack(real_conf_list, dim=2)
        self.real_out_vid = torch.stack(real_out_img_list, dim=2)
        self.real_warped_vid = torch.stack(real_warped_img_list, dim=2)
        # reference images are the same for different time steps, just pick the final one
        self.ref_img_fea = generated["bottle_neck_feat"].clone().detach()

        if self.is_train:
            if self.use_residual_flow:
                h, w, = H//4, W//4
                identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
                self.loss = self.diffusion(torch.cat((self.real_vid_grid - identity_grid,
                                                      self.real_vid_conf*2-1), dim=1),
                                           self.ref_img_fea,
                                           self.ref_text)
            else:
                self.loss = self.diffusion(torch.cat((self.real_vid_grid,
                                                      self.real_vid_conf*2-1), dim=1),
                                           self.ref_img_fea,
                                           self.ref_text)
            with torch.no_grad():
                fake_out_img_list = []
                fake_warped_img_list = []
                pred = self.diffusion.pred_x0
                if self.use_residual_flow:
                    self.fake_vid_grid = pred[:, :2, :, :, :] + identity_grid
                else:
                    self.fake_vid_grid = pred[:, :2, :, :, :]
                self.fake_vid_conf = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
                for idx in range(nf):
                    fake_grid = self.fake_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1)
                    fake_conf = self.fake_vid_conf[:, :, idx, :, :]
                    # predict fake out image and fake warped image
                    generated = self.generator.forward_with_flow(source_image=self.ref_img,
                                                                 optical_flow=fake_grid,
                                                                 occlusion_map=fake_conf)
                    fake_out_img_list.append(generated["prediction"])
                    fake_warped_img_list.append(generated["deformed"])
                self.fake_out_vid = torch.stack(fake_out_img_list, dim=2)
                self.fake_warped_vid = torch.stack(fake_warped_img_list, dim=2)
                self.rec_loss = nn.L1Loss()(self.real_vid, self.fake_out_vid)
                self.rec_warp_loss = nn.L1Loss()(self.real_vid, self.fake_warped_vid)

    def optimize_parameters(self):
        self.forward()
        self.optimizer_diff.zero_grad()
        if self.only_use_flow:
            self.loss.backward()
        else:
            (self.loss + self.rec_loss + self.rec_warp_loss).backward()
        self.optimizer_diff.step()

    def sample_one_video(self, cond_scale):
        self.sample_img_fea = self.generator.compute_fea(self.sample_img)
        # if cond_scale = 1.0, not using unconditional model
        pred = self.diffusion.sample(self.sample_img_fea, cond=self.sample_text,
                                     batch_size=1, cond_scale=cond_scale)
        if self.use_residual_flow:
            b, _, nf, h, w = pred[:, :2, :, :, :].size()
            identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
            self.sample_vid_grid = pred[:, :2, :, :, :] + identity_grid
        else:
            self.sample_vid_grid = pred[:, :2, :, :, :]
        self.sample_vid_conf = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
        nf = self.sample_vid_grid.size(2)
        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            for idx in range(nf):
                sample_grid = self.sample_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1)
                sample_conf = self.sample_vid_conf[:, :, idx, :, :]
                # predict fake out image and fake warped image
                generated = self.generator.forward_with_flow(source_image=self.sample_img,
                                                             optical_flow=sample_grid,
                                                             occlusion_map=sample_conf)
                sample_out_img_list.append(generated["prediction"])
                sample_warped_img_list.append(generated["deformed"])
        self.sample_out_vid = torch.stack(sample_out_img_list, dim=2)
        self.sample_warped_vid = torch.stack(sample_warped_img_list, dim=2)

    def set_train_input(self, ref_img, real_vid, ref_text):
        self.ref_img = ref_img.cuda()
        self.real_vid = real_vid.cuda()
        self.ref_text = ref_text

    def set_sample_input(self, sample_img, sample_text):
        self.sample_img = sample_img.cuda()
        self.sample_text = sample_text

    def print_learning_rate(self):
        lr = self.optimizer_diff.param_groups[0]['lr']
        assert lr > 0
        print('lr= %.7f' % lr)

    def get_grid(self, b, nf, H, W, normalize=True):
        if normalize:
            h_range = torch.linspace(-1, 1, H)
            w_range = torch.linspace(-1, 1, W)
        else:
            h_range = torch.arange(0, H)
            w_range = torch.arange(0, W)
        grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b, 1, 1, 1).flip(3).float()  # flip h,w to x,y
        return grid.permute(0, 3, 1, 2).unsqueeze(dim=2).repeat(1, 1, nf, 1, 1)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    bs = 5
    img_size = 64
    num_frames = 40
    ref_text = ["play basketball"] * bs
    ref_img = torch.rand((bs, 3, img_size, img_size), dtype=torch.float32)
    real_vid = torch.rand((bs, 3, num_frames, img_size, img_size), dtype=torch.float32)
    model = FlowDiffusion(use_residual_flow=False,
                          sampling_timesteps=10,
                          img_size=16,
                          config_pth="/workspace/code/CVPR23_LFDM/config/mug128.yaml",
                          pretrained_pth="")
    model.cuda()
    # model.train()
    # model.set_train_input(ref_img=ref_img, real_vid=real_vid, ref_text=ref_text)
    # model.optimize_parameters()
    model.eval()
    model.set_sample_input(sample_img=ref_img, sample_text=ref_text)
    model.sample_one_video(cond_scale=1.0)



