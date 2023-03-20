LFDM
=====
The pytorch implementation of our CVPR 2023 paper "Conditional Image-to-Video Generation with Latent Flow Diffusion Models"

This repository is still under development.

<div align=center><img src="architecture.png" width="915px" height="306px"/></div>

Updates
-----
[Updated on 03/20/2023] Released a testing demo for MUG dataset.

Example Videos
------
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
|MUG|LFAE| random         |https://drive.google.com/file/d/1dRn1wl5TUaZJiiDpIQADt1JJ0_q36MVG/view?usp=share_link|
|MUG|DM| random         |   https://drive.google.com/file/d/1lPVIT_cXXeOVogKLhD9fAT4k1Brd_HHn/view?usp=share_link |

Demo
-----
**MUG Dataset**

1. Install required dependencies. Here we use Python 3.7.10 and Pytorch 1.12.1, etc.
2. Set the paths in the code files and config files if needed. The pretrained models for MUG dataset have released. 
3. Run `python -u demo/demo_mug.py`

For questions with the code, please feel free to open an issue or contact me: homerhm.ni@gmail.com

Acknowledgement
----
Part of our code was borrowed from [MRAA](https://github.com/snap-research/articulated-animation), [VDM](https://github.com/lucidrains/video-diffusion-pytorch), and [LDM](https://github.com/CompVis/latent-diffusion). We thank the authors of these repositories for their valuable implementations.

