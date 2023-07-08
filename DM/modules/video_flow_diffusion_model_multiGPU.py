# based on video_flow_diffusion_model.py
# use diffusion model to generate pseudo ground truth flow volume based on RegionMM
# 3D noise to 3D flow
# flow size: 2*32*32*40
# enable multiple GPU

import os
import torch
import torch.nn as nn
from LFAE.modules.generator import Generator
from LFAE.modules.bg_motion_predictor import BGMotionPredictor
from LFAE.modules.region_predictor import RegionPredictor
from DM.modules.video_flow_diffusion_multiGPU import Unet3D, GaussianDiffusion
import yaml
from sync_batchnorm import DataParallelWithCallback
from DM.modules.text import tokenize, bert_embed


class FlowDiffusion(nn.Module):
    def __init__(self, img_size=32, num_frames=40, sampling_timesteps=250,
                 null_cond_prob=0.1,
                 ddim_sampling_eta=1.,
                 dim_mults=(1, 2, 4, 8),
                 is_train=True,
                 use_residual_flow=False,
                 learn_null_cond=False,
                 use_deconv=True,
                 padding_mode="zeros",
                 pretrained_pth="",
                 config_pth=""):
        super(FlowDiffusion, self).__init__()
        self.use_residual_flow = use_residual_flow

        checkpoint = torch.load(pretrained_pth)
        with open(config_pth) as f:
            config = yaml.safe_load(f)

        self.generator = Generator(num_regions=config['model_params']['num_regions'],
                                   num_channels=config['model_params']['num_channels'],
                                   revert_axis_swap=config['model_params']['revert_axis_swap'],
                                   **config['model_params']['generator_params']).cuda()
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator.eval()
        self.set_requires_grad(self.generator, False)

        self.region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                                num_channels=config['model_params']['num_channels'],
                                                estimate_affine=config['model_params']['estimate_affine'],
                                                **config['model_params']['region_predictor_params']).cuda()
        self.region_predictor.load_state_dict(checkpoint['region_predictor'])
        self.region_predictor.eval()
        self.set_requires_grad(self.region_predictor, False)

        self.bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                              **config['model_params']['bg_predictor_params'])
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
            timesteps=1000,  # number of steps
            loss_type='l2',  # L1 or L2
            use_dynamic_thres=True,
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta
        )

        # training
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()

    def forward(self, real_vid, ref_img, ref_text):
        # compute pseudo ground-truth flow
        b, _, nf, H, W = real_vid.size()

        real_grid_list = []
        real_conf_list = []
        real_out_img_list = []
        real_warped_img_list = []
        output_dict = {}
        with torch.no_grad():
            source_region_params = self.region_predictor(ref_img)
            for idx in range(nf):
                driving_region_params = self.region_predictor(real_vid[:, :, idx, :, :])
                bg_params = self.bg_predictor(ref_img, real_vid[:, :, idx, :, :])
                generated = self.generator(ref_img, source_region_params=source_region_params,
                                           driving_region_params=driving_region_params, bg_params=bg_params)
                generated.update({'source_region_params': source_region_params,
                                  'driving_region_params': driving_region_params})
                real_grid_list.append(generated["optical_flow"].permute(0, 3, 1, 2))
                # normalized occlusion map
                real_conf_list.append(generated["occlusion_map"])
                real_out_img_list.append(generated["prediction"])
                real_warped_img_list.append(generated["deformed"])
        output_dict["real_vid_grid"] = torch.stack(real_grid_list, dim=2)
        output_dict["real_vid_conf"] = torch.stack(real_conf_list, dim=2)
        output_dict["real_out_vid"] = torch.stack(real_out_img_list, dim=2)
        output_dict["real_warped_vid"] = torch.stack(real_warped_img_list, dim=2)
        # reference images are the same for different time steps, just pick the final one
        ref_img_fea = generated["bottle_neck_feat"].clone().detach()

        if self.is_train:
            if self.use_residual_flow:
                h, w, = H // 4, W // 4
                identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
                output_dict["loss"], output_dict["null_cond_mask"] = self.diffusion(
                    torch.cat((output_dict["real_vid_grid"] - identity_grid,
                               output_dict["real_vid_conf"] * 2 - 1), dim=1),
                    ref_img_fea,
                    ref_text)
            else:
                output_dict["loss"], output_dict["null_cond_mask"] = self.diffusion(
                    torch.cat((output_dict["real_vid_grid"],
                               output_dict["real_vid_conf"] * 2 - 1), dim=1),
                    ref_img_fea,
                    ref_text)
            with torch.no_grad():
                fake_out_img_list = []
                fake_warped_img_list = []
                pred = self.diffusion.pred_x0
                if self.use_residual_flow:
                    output_dict["fake_vid_grid"] = pred[:, :2, :, :, :] + identity_grid
                else:
                    output_dict["fake_vid_grid"] = pred[:, :2, :, :, :]
                output_dict["fake_vid_conf"] = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
                for idx in range(nf):
                    fake_grid = output_dict["fake_vid_grid"][:, :, idx, :, :].permute(0, 2, 3, 1)
                    fake_conf = output_dict["fake_vid_conf"][:, :, idx, :, :]
                    # predict fake out image and fake warped image
                    generated = self.generator.forward_with_flow(source_image=ref_img,
                                                                 optical_flow=fake_grid,
                                                                 occlusion_map=fake_conf)
                    fake_out_img_list.append(generated["prediction"])
                    fake_warped_img_list.append(generated["deformed"])
                output_dict["fake_out_vid"] = torch.stack(fake_out_img_list, dim=2)
                output_dict["fake_warped_vid"] = torch.stack(fake_warped_img_list, dim=2)
                output_dict["rec_loss"] = nn.L1Loss(reduce=False)(real_vid, output_dict["fake_out_vid"])
                output_dict["rec_warp_loss"] = nn.L1Loss(reduce=False)(real_vid, output_dict["fake_warped_vid"])

        return output_dict

    def sample_one_video(self, sample_img, sample_text, cond_scale):
        output_dict = {}
        sample_img_fea = self.generator.compute_fea(sample_img)
        bs = sample_img_fea.size(0)
        # if cond_scale = 1.0, not using unconditional model
        pred = self.diffusion.sample(sample_img_fea, cond=sample_text,
                                     batch_size=bs, cond_scale=cond_scale)
        if self.use_residual_flow:
            b, _, nf, h, w = pred[:, :2, :, :, :].size()
            identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
            output_dict["sample_vid_grid"] = pred[:, :2, :, :, :] + identity_grid
        else:
            output_dict["sample_vid_grid"] = pred[:, :2, :, :, :]
        output_dict["sample_vid_conf"] = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
        nf = output_dict["sample_vid_grid"].size(2)
        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            for idx in range(nf):
                sample_grid = output_dict["sample_vid_grid"][:, :, idx, :, :].permute(0, 2, 3, 1)
                sample_conf = output_dict["sample_vid_conf"][:, :, idx, :, :]
                # predict fake out image and fake warped image
                generated = self.generator.forward_with_flow(source_image=sample_img,
                                                             optical_flow=sample_grid,
                                                             occlusion_map=sample_conf)
                sample_out_img_list.append(generated["prediction"])
                sample_warped_img_list.append(generated["deformed"])
        output_dict["sample_out_vid"] = torch.stack(sample_out_img_list, dim=2)
        output_dict["sample_warped_vid"] = torch.stack(sample_warped_img_list, dim=2)
        return output_dict

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    bs = 10
    img_size = 128
    num_frames = 40
    ref_text = ["play basketball"] * bs
    ref_img = torch.rand((bs, 3, img_size, img_size), dtype=torch.float32).cuda()
    real_vid = torch.rand((bs, 3, num_frames, img_size, img_size), dtype=torch.float32).cuda()
    model = FlowDiffusion(use_residual_flow=False, sampling_timesteps=10, dim_mults=(1, 2, 4, 8, 16))
    model.cuda()
    # embedding ref_text
    cond = bert_embed(tokenize(ref_text), return_cls_repr=model.diffusion.text_use_bert_cls).cuda()
    model = DataParallelWithCallback(model)
    output_dict = model.forward(real_vid=real_vid, ref_img=ref_img, ref_text=cond)
    model.module.sample_one_video(sample_img=ref_img[0].unsqueeze(dim=0),
                                  sample_text=[ref_text[0]],
                                  cond_scale=1.0)
