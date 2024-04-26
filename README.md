!!! Check out our new CVPR 2024 [paper](https://arxiv.org/abs/2404.16306) designed for text-conditioned image-to-video generation

LFDM
=====
The pytorch implementation of our CVPR 2023 paper [Conditional Image-to-Video Generation with Latent Flow Diffusion Models](https://arxiv.org/abs/2303.13744).

<div align=center><img src="architecture.png" width="915px" height="306px"/></div>

Updates
-----
[Updated on 07/08/2023] Added multi-GPU training codes for MHAD dataset.

[Updated on 05/12/2023] Released a testing demo for NATOPS dataset.

[Updated on 03/31/2023] Added the illustration of training a LFDM for NATOPS dataset.

[Updated on 03/27/2023] Added the illustration of training a LFDM for MHAD dataset.

[Updated on 03/27/2023] Released a testing demo for MHAD dataset.

[Updated on 03/26/2023] Added the illustration of training a LFDM for MUG dataset.

[Updated on 03/26/2023] Now our paper is available on [arXiv](https://arxiv.org/abs/2303.13744).

[Updated on 03/20/2023] Released a testing demo for MUG dataset.

Example Videos
------
All the subjects of the following videos are *unseen* during the training. 

Some generated video results on MUG dataset.

<div align=center>
<img src="examples/mug.gif" width="500" height="276"/>
</div>

Some generated video results on MHAD dataset.

<div align=center>
<img src="examples/mhad1.gif" width="500" height="530"/>
</div>
<div align=center>
<img src="examples/mhad2.gif" width="500" height="416"/>
</div>

Some generated video results on NATOPS dataset.

<div align=center>
<img src="examples/natops.gif" width="500" height="525"/>
</div>

Applied LFDM trained on MUG to FaceForensics dataset.

<div align=center>
<img src="examples/new_domain_grid.gif" width="400" height="523"/>
</div>

Pretrained Models
-----

|Dataset|Model| Frame Sampling |Link (Google Drive)|
|-------|------|----------------|-----|
|MUG|LFAE| -         |https://drive.google.com/file/d/1dRn1wl5TUaZJiiDpIQADt1JJ0_q36MVG/view?usp=share_link|
|MUG|DM| very_random         |   https://drive.google.com/file/d/1lPVIT_cXXeOVogKLhD9fAT4k1Brd_HHn/view?usp=share_link |
|MHAD|LFAE|-|https://drive.google.com/file/d/1AVtpKbzqsXdIK-_vHUuQQIGx6Wa5PxS0/view?usp=share_link|
|MHAD|DM|random|https://drive.google.com/file/d/1BoFPQAeOuHE5wt7h-chhYAO-dU0B1p2y/view?usp=share_link|
|NATOPS|LFAE|-|https://drive.google.com/file/d/10iyzoYqSwzQ3fZgb6oh3Uay-P7k2A12s/view?usp=share_link|
|NATOPS|DM|random|https://drive.google.com/file/d/1lSLSzS_KyGvJ7dW3l5hLJLR9k2k8LoU3/view?usp=share_link|

Demo
-----
**MUG Dataset**

1. Install required dependencies. Here we use Python 3.7.10 and Pytorch 1.12.1, etc.
2. Run `python -u demo/demo_mug.py` to generate the example videos. Please set the paths in the code files and config file `config/mug128.yaml` if needed. The pretrained models for MUG dataset have released. 

**MHAD Dataset**

1. Install required dependencies. Here we use Python 3.7.10 and Pytorch 1.12.1, etc.
2. Run `python -u demo/demo_mhad.py` to generate the example videos. Please set the paths in the code files and config file `config/mhad128.yaml` if needed. The pretrained models for MHAD dataset have released. 

**NATOPS Dataset**

1. Install required dependencies. Here we use Python 3.7.10 and Pytorch 1.12.1, etc.
2. Run `python -u demo/demo_natops.py` to generate the example videos. Please set the paths in the code files and config file `config/natops128.yaml` if needed. The pretrained models for NATOPS dataset have released. 

Training LFDM
----
The training of our LFDM includes two stages: 1. train a latent flow autoencoder (LFAE) in an unsupervised fashion. To accelerate the training, we initialize LFAE with the pretrained models provided by MRAA, which can be found in their [github](https://github.com/snap-research/articulated-animation/tree/db2c2135273f601a370e2b62754f9bb56cfd25d5/checkpoints); 2. train a diffusion model (DM) on the latent space of LFAE.

**MUG Dataset**

1. Download MUG dataset from their [website](https://mug.ee.auth.gr/fed/). 
2. Install required dependencies. Here we use Python 3.7.10 and Pytorch 1.12.1, etc.
3. Split the train/test set. You may use the same split as ours, which can be found in `preprocessing/preprocess_MUG.py`.
4. Run `python -u LFAE/run_mug.py` to train the LFAE. Please set the paths and config file `config/mug128.yaml` if needed. 
5. Once LFAE is trained, you may measure its self-reconstruction performance by running `python -u LFAE/test_flowautoenc_mug.py`.
6. Run `python -u DM/train_video_flow_diffusion_mug.py` to train the DM. Please set the paths and config file `config/mug128.yaml` if needed. 
7. Once DM is trained, you may test its generation performance by running `python -u DM/test_video_flow_diffusion_mug.py`.

**MHAD Dataset**

1. Download MHAD dataset from their [website](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html). 
2. Install required dependencies. Here we use Python 3.7.10 and Pytorch 1.12.1, etc.
3. Crop the video frames and split the train/test set. You may use the same cropping method and split as ours, which can be found in `preprocessing/preprocess_MHAD.py`.
4. Run `python -u LFAE/run_mhad.py` to train the LFAE. Please set the paths and config file `config/mhad128.yaml` if needed. 
5. Once LFAE is trained, you may measure its self-reconstruction performance by running `python -u LFAE/test_flowautoenc_mhad.py`.
6. Run `python -u DM/train_video_flow_diffusion_mhad.py` to train the DM. Please set the paths and config file `config/mhad128.yaml` if needed. 
7. Once DM is trained, you may test its generation performance by running `python -u DM/test_video_flow_diffusion_mhad.py`.

**NATOPS Dataset**

1. Download NATOPS dataset from their [website](https://github.com/yalesong/natops). 
2. Install required dependencies. Here we use Python 3.7.10 and Pytorch 1.12.1, etc.
3. Segment the video and split the train/test set. You may use the same segmenting method and split as ours, which can be found in `preprocessing/preprocess_NATOPS.py`.
4. Run `python -u LFAE/run_natops.py` to train the LFAE. Please set the paths and config file `config/natops128.yaml` if needed. 
5. Once LFAE is trained, you may measure its self-reconstruction performance by running `python -u LFAE/test_flowautoenc_natops.py`.
6. Run `python -u DM/train_video_flow_diffusion_natops.py` to train the DM. Please set the paths and config file `config/natops128.yaml` if needed. 
7. Once DM is trained, you may test its generation performance by running `python -u DM/test_video_flow_diffusion_natops.py`.

Citing LFDM
-------
If you find our approaches useful in your research, please consider citing:
```
@inproceedings{ni2023conditional,
  title={Conditional Image-to-Video Generation with Latent Flow Diffusion Models},
  author={Ni, Haomiao and Shi, Changhao and Li, Kai and Huang, Sharon X and Min, Martin Renqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18444--18455},
  year={2023}
}
```

For questions with the code, please feel free to open an issue or contact me: homerhm.ni@gmail.com

Acknowledgement
----
Part of our code was borrowed from [MRAA](https://github.com/snap-research/articulated-animation), [VDM](https://github.com/lucidrains/video-diffusion-pytorch), and [LDM](https://github.com/CompVis/latent-diffusion). We thank the authors of these repositories for their valuable implementations.

