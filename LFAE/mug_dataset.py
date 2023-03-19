# Created on 08/08/2022
# build MUG dataset for RegionMM

import os

import torch.utils.data
import imageio
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
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


# this is just for training
class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=128, id_sampling=False,
                 pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.frame_shape = frame_shape
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        train_ID = ['008', '017', '021', '028', '030', '031', '034', '036', '037',
                    '038', '039', '042', '043', '044', '045', '055', '060', '061',
                    '062', '063', '071', '075', '076', '077', '083', '084']
        session_ID = ["002", "003", "049"]
        expression = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.videos = []
        for video_name in train_ID:
            if video_name not in session_ID:
                for exp_name in expression:
                    cur_video_dir_path = os.path.join(root_dir, video_name, exp_name)
                    if os.path.exists(cur_video_dir_path):
                        cur_video_name_list = os.listdir(cur_video_dir_path)
                        cur_video_name_list.sort()
                        for cur_video_name in cur_video_name_list:
                            cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                            self.videos.append(cur_video_path)
            else:
                for session_name in ["session0", "session0"]:
                    for exp_name in expression:
                        cur_video_dir_path = os.path.join(root_dir, video_name, session_name, exp_name)
                        if os.path.exists(cur_video_dir_path):
                            cur_video_name_list = os.listdir(cur_video_dir_path)
                            cur_video_name_list.sort()
                            for cur_video_name in cur_video_name_list:
                                cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                                self.videos.append(cur_video_path)

        self.transform = AllAugmentationTransform(**augmentation_params)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.id_sampling:
            # TODO
            name = self.videos[idx]
            try:
                sub_name = name.split("_")[1]
                path = np.random.choice(glob.glob(os.path.join(self.root_dir, 'a*_%s_t*' % sub_name)))
            except ValueError:
                raise ValueError("File formatting is not correct for id_sampling=True. "
                                 "Change file formatting, or set id_sampling=False.")
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = "_".join(path.split("/")[-3:]) if "session" not in path else "_".join(path.split("/")[-4:])

        frames = os.listdir(path)
        frames.sort()
        # remove the final avi file
        frames = [x for x in frames if x.endswith("jpg") or x.endswith("png")]
        num_frames = len(frames)

        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

        resize_fn = partial(resize, desired_size=self.frame_shape, interpolation=cv2.INTER_AREA)

        if type(frames[0]) is bytes:
            frame_names = [frames[idx].decode('utf-8') for idx in frame_idx]
        else:
            frame_names = [frames[idx] for idx in frame_idx]

        video_array = [resize_fn(imageio.imread(os.path.join(path, x))) for x in frame_names]

        # video_array = [img_as_float32(x) for x in video_array]

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


# added by nhm
class MUG_pair_test(Dataset):
    def __init__(self, data_dir, image_size=128):
        super(MUG_pair_test, self).__init__()
        self.image_size = image_size
        test_ID = ['001', '002', '006', '007', '010', '013', '014', '020', '027', '032',
                   '033', '040', '046', '048', '049', '052', '064', '065', '066', '070',
                   '072', '073', '074', '078', '079', '082']
        session_ID = ["002", "003", "049"]
        expression = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

        self.video_path_list = []
        for video_name in test_ID:
            if video_name not in session_ID:
                for exp_name in expression:
                    cur_video_dir_path = os.path.join(data_dir, video_name, exp_name)
                    if os.path.exists(cur_video_dir_path):
                        cur_video_name_list = os.listdir(cur_video_dir_path)
                        cur_video_name_list.sort()
                        for cur_video_name in cur_video_name_list:
                            cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                            self.video_path_list.append(cur_video_path)
            else:
                for session_name in ["session0", "session0"]:
                    for exp_name in expression:
                        cur_video_dir_path = os.path.join(data_dir, video_name, session_name, exp_name)
                        if os.path.exists(cur_video_dir_path):
                            cur_video_name_list = os.listdir(cur_video_dir_path)
                            cur_video_name_list.sort()
                            for cur_video_name in cur_video_name_list:
                                cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                                self.video_path_list.append(cur_video_path)

    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        video_name = "_".join(video_path.split("/")[-3:]) if "session" not in video_path \
            else "_".join(video_path.split("/")[-4:])
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        sample_path_list = [frame_path_list[1], frame_path_list[len(frame_path_list)//2]]
        src_img_arr = imageio.imread(sample_path_list[0])
        tar_img_arr = imageio.imread(sample_path_list[1])

        src_img_name = os.path.basename(sample_path_list[0])
        tar_img_name = os.path.basename(sample_path_list[1])

        src_img_arr = np.asarray(src_img_arr, np.float32)
        tar_img_arr = np.asarray(tar_img_arr, np.float32)

        src_img_arr = resize(src_img_arr, desired_size=self.image_size, interpolation=cv2.INTER_AREA)
        tar_img_arr = resize(tar_img_arr, desired_size=self.image_size, interpolation=cv2.INTER_AREA)

        src_img_arr = src_img_arr.transpose((2, 0, 1))
        tar_img_arr = tar_img_arr.transpose((2, 0, 1))

        src_img_arr = src_img_arr/255.0
        tar_img_arr = tar_img_arr/255.0

        return src_img_arr, tar_img_arr, src_img_name, tar_img_name, video_name


if __name__ == "__main__":
    import yaml
    config_file = "/workspace/code/nec-project/articulated-animation-dgx/config/mug128.yaml"
    with open(config_file) as f:
        config = yaml.load(f)
    root_dir = "/data/hfn5052/text2motion/MUG"
    # dataset = FramesDataset(**config['dataset_params'])
    dataset = MUG_pair_test(data_dir=root_dir)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=100)
    for i_iter, batch in enumerate(test_loader):
        print(i_iter)
