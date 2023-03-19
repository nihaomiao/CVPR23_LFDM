# utilize RegionMM to design a flow auto-encoder

import torch
import torch.nn as nn
from LFAE.modules.generator import Generator
from LFAE.modules.bg_motion_predictor import BGMotionPredictor
from LFAE.modules.region_predictor import RegionPredictor
import yaml


# based on RegionMM
class FlowAE(nn.Module):
    def __init__(self, is_train=False,
                 config_pth="/workspace/code/CVPR23_LFDM/config/mug128.yaml"):
        super(FlowAE, self).__init__()

        with open(config_pth) as f:
            config = yaml.safe_load(f)

        self.generator = Generator(num_regions=config['model_params']['num_regions'],
                                   num_channels=config['model_params']['num_channels'],
                                   revert_axis_swap=config['model_params']['revert_axis_swap'],
                                   **config['model_params']['generator_params']).cuda()
        self.region_predictor = RegionPredictor(num_regions=config['model_params']['num_regions'],
                                                num_channels=config['model_params']['num_channels'],
                                                estimate_affine=config['model_params']['estimate_affine'],
                                                **config['model_params']['region_predictor_params']).cuda()
        self.bg_predictor = BGMotionPredictor(num_channels=config['model_params']['num_channels'],
                                              **config['model_params']['bg_predictor_params'])

        self.is_train = is_train

        self.ref_img = None
        self.dri_img = None
        self.generated = None

    def forward(self):
        source_region_params = self.region_predictor(self.ref_img)
        self.driving_region_params = self.region_predictor(self.dri_img)

        bg_params = self.bg_predictor(self.ref_img, self.dri_img)
        self.generated = self.generator(self.ref_img, source_region_params=source_region_params,
                                        driving_region_params=self.driving_region_params, bg_params=bg_params)
        self.generated.update({'source_region_params': source_region_params,
                               'driving_region_params': self.driving_region_params})

    def set_train_input(self, ref_img, dri_img):
        self.ref_img = ref_img.cuda()
        self.dri_img = dri_img.cuda()


if __name__ == "__main__":
    # default image size is 128
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ref_img = torch.rand((5, 3, 128, 128), dtype=torch.float32)
    dri_img = torch.rand((5, 3, 128, 128), dtype=torch.float32)
    model = FlowAE(is_train=True).cuda()
    model.train()
    model.set_train_input(ref_img=ref_img, dri_img=dri_img)
    model.forward()

