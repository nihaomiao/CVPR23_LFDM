import cv2
import os

import requests
import torch
import torch.nn.functional as F
import torch.distributed as dist
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import flow_vis
import cv2


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def grid2fig(warped_grid, grid_size=32, img_size=256):
    dpi = 1000
    # plt.ioff()
    h_range = torch.linspace(-1, 1, grid_size)
    w_range = torch.linspace(-1, 1, grid_size)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2)
    flow_uv = grid.cpu().data.numpy()
    fig, ax = plt.subplots()
    grid_x, grid_y = warped_grid[..., 0], warped_grid[..., 1]
    plot_grid(flow_uv[..., 0], flow_uv[..., 1], ax=ax, color="lightgrey")
    plot_grid(grid_x, grid_y, ax=ax, color="C0")
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.set_size_inches(img_size/100, img_size/100)
    fig.set_dpi(100)
    out = fig2data(fig)[:, :, :3]
    plt.close()
    plt.cla()
    plt.clf()
    return out


def flow2fig(warped_grid, id_grid, grid_size=32, img_size=128):
    # h_range = torch.linspace(-1, 1, grid_size)
    # w_range = torch.linspace(-1, 1, grid_size)
    # id_grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2)
    warped_flow = warped_grid - id_grid
    img = flow_vis.flow_to_color(warped_flow)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def conf2fig(conf, img_size=128):
    conf = F.interpolate(conf.unsqueeze(dim=0), size=img_size).data.cpu().numpy()
    conf = np.transpose(conf, [0, 2, 3, 1])
    conf = np.array(conf[0, :, :, 0]*255, dtype=np.uint8)
    return conf


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def resize(im, desired_size, interpolation):
    old_size = im.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple(int(x*ratio) for x in old_size)

    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=interpolation)
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


def resample(image, flow):
    r"""Resamples an image using the provided flow.

    Args:
        image (NxCxHxW tensor) : Image to resample.
        flow (Nx2xHxW tensor) : Optical flow to resample the image.
    Returns:
        output (NxCxHxW tensor) : Resampled image.
    """
    assert flow.shape[1] == 2
    b, c, h, w = image.size()
    grid = get_grid(b, (h, w))
    flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
    final_grid = (grid + flow).permute(0, 2, 3, 1)
    try:
        output = F.grid_sample(image, final_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
    except Exception:
        output = F.grid_sample(image, final_grid, mode='bilinear',
                               padding_mode='border')
    return output


def get_grid(batchsize, size, minval=-1.0, maxval=1.0):
    r"""Get a grid ranging [-1, 1] of 2D/3D coordinates.

    Args:
        batchsize (int) : Batch size.
        size (tuple) : (height, width) or (depth, height, width).
        minval (float) : minimum value in returned grid.
        maxval (float) : maximum value in returned grid.
    Returns:
        t_grid (4D tensor) : Grid of coordinates.
    """
    if len(size) == 2:
        rows, cols = size
    elif len(size) == 3:
        deps, rows, cols = size
    else:
        raise ValueError('Dimension can only be 2 or 3.')
    x = torch.linspace(minval, maxval, cols)
    x = x.view(1, 1, 1, cols)
    x = x.expand(batchsize, 1, rows, cols)

    y = torch.linspace(minval, maxval, rows)
    y = y.view(1, 1, rows, 1)
    y = y.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([x, y], dim=1)

    if len(size) == 3:
        z = torch.linspace(minval, maxval, deps)
        z = z.view(1, 1, deps, 1, 1)
        z = z.expand(batchsize, 1, deps, rows, cols)

        t_grid = t_grid.unsqueeze(2).expand(batchsize, 2, deps, rows, cols)
        t_grid = torch.cat([t_grid, z], dim=1)

    t_grid.requires_grad = False
    return t_grid.to('cuda')


def get_checkpoint(checkpoint_path, url=''):
    r"""Get the checkpoint path. If it does not exist yet, download it from
    the url.

    Args:
        checkpoint_path (str): Checkpoint path.
        url (str): URL to download checkpoint.
    Returns:
        (str): Full checkpoint path.
    """
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = os.getcwd()
    save_dir = os.path.join(os.environ['TORCH_HOME'], 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    full_checkpoint_path = os.path.join(save_dir, checkpoint_path)
    if not os.path.exists(full_checkpoint_path):
        os.makedirs(os.path.dirname(full_checkpoint_path), exist_ok=True)
        if is_master():
            print('Download {}'.format(url))
            download_file_from_google_drive(url, full_checkpoint_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return full_checkpoint_path


def download_file_from_google_drive(file_id, destination):
    r"""Download a file from the google drive by using the file ID.

    Args:
        file_id: Google drive file ID
        destination: Path to save the file.

    Returns:

    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    r"""Get confirm token

    Args:
        response: Check if the file exists.

    Returns:

    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    r"""Save response content

    Args:
        response:
        destination: Path to save the file.

    Returns:

    """
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0

