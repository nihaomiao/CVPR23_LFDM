# build NATOPS dataset for RegionMM

import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
from augmentation import AllAugmentationTransform
from functools import partial
import cv2
import matplotlib.pyplot as plt


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


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)]
        if frame_shape is not None:
            video_array = np.array([resize(frame, frame_shape) for frame in video_array])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if frame_shape is None:
            raise ValueError('Frame shape can not be None for stacked png format.')

        frame_shape = tuple(frame_shape)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape + (3, ))
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = mimread(name)
        if len(video[0].shape) == 2:
            video = [gray2rgb(frame) for frame in video]
        if frame_shape is not None:
            video = np.array([resize(frame, frame_shape) for frame in video])
        video = np.array(video)
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


# this is just for training
class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=128, use_crop=True,
                 pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.frame_shape = frame_shape
        self.pairs_list = pairs_list
        self.use_crop = use_crop
        if self.use_crop:
            self.y_min, self.y_max, self.x_min, self.x_max = 10, 239, 30, 290
            print("use crop box:", self.y_min, self.y_max, self.x_min, self.x_max)

        video_name_list = os.listdir(root_dir)
        video_name_list.sort()
        train_ID = [3, 4, 8, 9, 12, 13, 15, 17, 19, 20]
        self.videos = []
        for video_name in video_name_list:
            sub_name = int(video_name[4:6])
            if sub_name in train_ID:
                self.videos.append(video_name)

        self.transform = AllAugmentationTransform(**augmentation_params)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        frames = os.listdir(path)
        num_frames = len(frames)

        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

        resize_fn = partial(resize, desired_size=self.frame_shape, interpolation=cv2.INTER_AREA)

        if type(frames[0]) is bytes:
            frame_names = [frames[idx].decode('utf-8') for idx in frame_idx]
        else:
            frame_names = [frames[idx] for idx in frame_idx]

        video_array = [io.imread(os.path.join(path, x)) for x in frame_names]
        if self.use_crop:
            video_array = [x[self.y_min:self.y_max, self.x_min:self.x_max, :] for x in video_array]
        video_array = [resize_fn(x) for x in video_array]

        video_array = self.transform(video_array)

        out = {}

        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')

        out['driving'] = driving.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['name'] = video_name
        out['frame'] = frame_names
        out['id'] = idx
        
        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


if __name__ == "__main__":
    pass
