import os
import random

import numpy as np
import torch
import torch.utils.data as data
import imageio
from misc import resize
import cv2
import torchvision.transforms.functional as F
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


class NATOPS(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=True, use_crop=True,
                 sampling="very_random"):
        super(NATOPS, self).__init__()
        self.mean = mean
        self.is_jitter = color_jitter
        self.sampling = sampling

        self.use_crop = use_crop
        if self.use_crop:
            self.y_min, self.y_max, self.x_min, self.x_max = 10, 239, 30, 290
            print("use crop box:", self.y_min, self.y_max, self.x_min, self.x_max)

        self.action_list = ["I Have Command",
                            "All Clear",
                            "Not Clear",
                            "Spread Wings",
                            "Fold Wings",
                            "Lock Wings",
                            "Up Hook",
                            "Down Hook",
                            "Remove Tiedowns",
                            "Remove Chocks",
                            "Insert Chocks",
                            "Move Ahead",
                            "Turn Left",
                            "Turn Right",
                            "Next Marshaller",
                            "Slow Down",
                            "Stop",
                            "Nosegear Steering",
                            "Hot Brakes",
                            "Brakes On",
                            "Brakes Off",
                            "Install Tiedowns",
                            "Fire",
                            "Cut Engine"]
        self.num_frames = num_frames
        self.image_size = image_size
        video_name_list = os.listdir(data_dir)
        video_name_list.sort()

        train_ID = [3, 4, 8, 9, 12, 13, 15, 17, 19, 20]
        train_video_name_list = []
        for video_name in video_name_list:
            sub_name = int(video_name[4:6])
            if sub_name in train_ID:
                train_video_name_list.append(video_name)

        self.video_path_list = [os.path.join(data_dir, video_name) for video_name in train_video_name_list]

    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        video_name = os.path.basename(video_path)
        action_idx = int(video_name[1:3])
        action_name = self.action_list[action_idx - 1]
        frame_name_list = os.listdir(video_path)
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        total_num_frames = len(frame_path_list)
        if total_num_frames >= self.num_frames:
            # uniform sampling
            if self.sampling == "uniform":
                sample_idx_list = np.linspace(start=0, stop=total_num_frames-1, num=self.num_frames, dtype=int)
            if self.sampling == "random":
                uniform_idx_list = np.linspace(start=0, stop=total_num_frames-1, num=self.num_frames, dtype=int)
                step_list = uniform_idx_list[1:] - uniform_idx_list[0:-1]
                sample_idx_list = uniform_idx_list.copy()
                for ii in range(1, self.num_frames - 1):
                    low = 1-step_list[ii-1]
                    high = +step_list[ii]
                    sample_idx_list[ii] = sample_idx_list[ii] + np.random.randint(low=low, high=high)
                sample_idx_list = np.sort(sample_idx_list)
                # compare = np.stack((uniform_idx_list, np.sort(sample_idx_list)))
        else:
            # simply repeat the final frame
            sample_idx_list = np.pad(list(range(total_num_frames)), (0, self.num_frames-total_num_frames), "edge")

        # very random sampling
        if self.sampling == "very_random":
            sample_idx_list = np.sort(np.random.choice(total_num_frames, self.num_frames, replace=True))
            # make the first frame to be 0
            sample_idx_list[0] = 0

        sample_frame_path_list = [frame_path_list[x] for x in sample_idx_list]
        # read image
        sample_frame_list = [imageio.imread(x) for x in sample_frame_path_list]
        # crop image
        if self.use_crop:
            sample_frame_list = [x[self.y_min:self.y_max, self.x_min:self.x_max, :] for x in sample_frame_list]
        # data augmentation
        sample_frame_list = [Image.fromarray(x) for x in sample_frame_list]
        if self.is_jitter:
            bright = 64. / 255
            contrast = 0.25
            sat = 0.25
            hue = 0.04
            bright_f = random.uniform(max(0, 1 - bright), 1 + bright)
            contrast_f = random.uniform(max(0, 1 - contrast), 1 + contrast)
            sat_f = random.uniform(max(0, 1 - sat), 1 + sat)
            hue_f = random.uniform(-hue, hue)
            sample_frame_list = [F.adjust_brightness(x, bright_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_contrast(x, contrast_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_saturation(x, sat_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_hue(x, hue_f) for x in sample_frame_list]
        sample_frame_list = [np.asarray(x, np.float32) for x in sample_frame_list]
        # resize to (image_size, image_size)
        sample_frame_list = [resize(x, self.image_size, interpolation=cv2.INTER_AREA) for x in sample_frame_list]
        sample_frame_list = [x - self.mean for x in sample_frame_list]
        sample_frame_list = [np.transpose(x, (2, 0, 1)) for x in sample_frame_list]
        sample_frame_list_npy = np.stack(sample_frame_list, axis=1)
        # change to float32
        sample_frame_list_npy = np.array(sample_frame_list_npy/255.0, dtype=np.float32)
        return sample_frame_list_npy, action_name, video_name


class NATOPS_test(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=False, use_crop=True):
        super(NATOPS_test, self).__init__()
        self.mean = mean
        self.is_jitter = color_jitter

        self.use_crop = use_crop
        if self.use_crop:
            self.y_min, self.y_max, self.x_min, self.x_max = 10, 239, 30, 290
            print("use crop box:", self.y_min, self.y_max, self.x_min, self.x_max)

        self.action_list = ["I Have Command",
                            "All Clear",
                            "Not Clear",
                            "Spread Wings",
                            "Fold Wings",
                            "Lock Wings",
                            "Up Hook",
                            "Down Hook",
                            "Remove Tiedowns",
                            "Remove Chocks",
                            "Insert Chocks",
                            "Move Ahead",
                            "Turn Left",
                            "Turn Right",
                            "Next Marshaller",
                            "Slow Down",
                            "Stop",
                            "Nosegear Steering",
                            "Hot Brakes",
                            "Brakes On",
                            "Brakes Off",
                            "Install Tiedowns",
                            "Fire",
                            "Cut Engine"]
        self.num_frames = num_frames
        self.image_size = image_size
        video_name_list = os.listdir(data_dir)
        video_name_list.sort()

        test_ID = [2, 5, 6, 7, 10, 11, 14, 16, 18]
        test_video_name_list = []
        for video_name in video_name_list:
            sub_name = int(video_name[4:6])
            if sub_name in test_ID:
                test_video_name_list.append(video_name)

        self.video_path_list = [os.path.join(data_dir, video_name) for video_name in test_video_name_list]

    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        video_name = os.path.basename(video_path)
        action_idx = int(video_name[1:3])
        action_name = self.action_list[action_idx - 1]
        frame_name_list = os.listdir(video_path)
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        total_num_frames = len(frame_path_list)
        if total_num_frames >= self.num_frames:
            # uniform sampling
            sample_idx_list = np.linspace(start=0, stop=total_num_frames-1, num=self.num_frames, dtype=int)
        else:
            # simply repeat the final frame
            sample_idx_list = np.pad(list(range(total_num_frames)), (0, self.num_frames-total_num_frames), "edge")
        sample_frame_path_list = [frame_path_list[x] for x in sample_idx_list]
        # read image
        sample_frame_list = [imageio.imread(x) for x in sample_frame_path_list]
        # crop image
        if self.use_crop:
            sample_frame_list = [x[self.y_min:self.y_max, self.x_min:self.x_max, :] for x in sample_frame_list]
        # data augmentation
        sample_frame_list = [Image.fromarray(x) for x in sample_frame_list]
        if self.is_jitter:
            bright = 64. / 255
            contrast = 0.25
            sat = 0.25
            hue = 0.04
            bright_f = random.uniform(max(0, 1 - bright), 1 + bright)
            contrast_f = random.uniform(max(0, 1 - contrast), 1 + contrast)
            sat_f = random.uniform(max(0, 1 - sat), 1 + sat)
            hue_f = random.uniform(-hue, hue)
            sample_frame_list = [F.adjust_brightness(x, bright_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_contrast(x, contrast_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_saturation(x, sat_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_hue(x, hue_f) for x in sample_frame_list]
        sample_frame_list = [np.asarray(x, np.float32) for x in sample_frame_list]
        sample_frame_list = [resize(x, self.image_size, interpolation=cv2.INTER_AREA) for x in sample_frame_list]
        sample_frame_list = [x - self.mean for x in sample_frame_list]
        sample_frame_list = [np.transpose(x, (2, 0, 1)) for x in sample_frame_list]
        sample_frame_list_npy = np.stack(sample_frame_list, axis=1)
        # change to float32
        sample_frame_list_npy = np.array(sample_frame_list_npy/255.0, dtype=np.float32)
        return sample_frame_list_npy, action_name, video_name


class NATOPS_gen(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=False, use_crop=True, sampling="very_random"):
        super(NATOPS_gen, self).__init__()
        self.sampling = sampling
        self.mean = mean
        self.is_jitter = color_jitter

        self.use_crop = use_crop
        if self.use_crop:
            self.y_min, self.y_max, self.x_min, self.x_max = 10, 239, 30, 290
            print("use crop box:", self.y_min, self.y_max, self.x_min, self.x_max)

        self.action_list = ["I Have Command",
                            "All Clear",
                            "Not Clear",
                            "Spread Wings",
                            "Fold Wings",
                            "Lock Wings",
                            "Up Hook",
                            "Down Hook",
                            "Remove Tiedowns",
                            "Remove Chocks",
                            "Insert Chocks",
                            "Move Ahead",
                            "Turn Left",
                            "Turn Right",
                            "Next Marshaller",
                            "Slow Down",
                            "Stop",
                            "Nosegear Steering",
                            "Hot Brakes",
                            "Brakes On",
                            "Brakes Off",
                            "Install Tiedowns",
                            "Fire",
                            "Cut Engine"]
        self.num_frames = num_frames
        self.image_size = image_size
        video_name_list = os.listdir(data_dir)
        video_name_list.sort()

        self.test_ID = [2, 5, 6, 7, 10, 11, 14, 16, 18]
        test_video_name_list = []
        for video_name in video_name_list:
            sub_name = int(video_name[4:6])
            if sub_name in self.test_ID:
                test_video_name_list.append(video_name)

        self.num_combs = len(self.test_ID) * len(self.action_list)

        self.video_path_list = [os.path.join(data_dir, video_name) for video_name in test_video_name_list]

        # group each video according to subject and expression
        self.video_dict = {}
        for comb_idx in range(self.num_combs):
            sub_idx = comb_idx % 9
            action_idx = comb_idx // 9
            sub_name = self.test_ID[sub_idx]
            action_name = self.action_list[action_idx]
            if sub_name not in self.video_dict.keys():
                self.video_dict[sub_name] = {}
            self.video_dict[sub_name][action_name] = []

        for video_path in self.video_path_list:
            video_name = os.path.basename(video_path)
            action_idx = int(video_name[1:3])
            action_name = self.action_list[action_idx - 1]
            sub_name = int(video_name[4:6])
            assert sub_name in self.test_ID
            assert action_name in self.action_list
            self.video_dict[sub_name][action_name].append(video_path)

    def __len__(self):
        return self.num_combs

    def __getitem__(self, index):
        sub_idx = index % 9
        action_idx = index // 9
        sub_name = self.test_ID[sub_idx]
        action_name = self.action_list[action_idx]
        video_path_list = self.video_dict[sub_name][action_name]
        assert len(video_path_list) > 0
        video_path = str(np.random.choice(video_path_list, size=1)[0])
        video_name = os.path.basename(video_path)
        frame_name_list = os.listdir(video_path)
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        total_num_frames = len(frame_path_list)
        if total_num_frames >= self.num_frames:
            # uniform sampling
            sample_idx_list = np.linspace(start=0, stop=total_num_frames-1, num=self.num_frames, dtype=int)
        else:
            # simply repeat the final frame
            sample_idx_list = np.pad(list(range(total_num_frames)), (0, self.num_frames-total_num_frames), "edge")

        # very random sampling
        if self.sampling == "very_random":
            sample_idx_list = np.sort(np.random.choice(total_num_frames, self.num_frames, replace=True))
            # make the first frame to be 0
            sample_idx_list[0] = 0

        sample_frame_path_list = [frame_path_list[x] for x in sample_idx_list]
        # read image
        sample_frame_list = [imageio.v2.imread(x) for x in sample_frame_path_list]
        # crop image
        if self.use_crop:
            sample_frame_list = [x[self.y_min:self.y_max, self.x_min:self.x_max, :] for x in sample_frame_list]
        # data augmentation
        sample_frame_list = [Image.fromarray(x) for x in sample_frame_list]
        if self.is_jitter:
            bright = 64. / 255
            contrast = 0.25
            sat = 0.25
            hue = 0.04
            bright_f = random.uniform(max(0, 1 - bright), 1 + bright)
            contrast_f = random.uniform(max(0, 1 - contrast), 1 + contrast)
            sat_f = random.uniform(max(0, 1 - sat), 1 + sat)
            hue_f = random.uniform(-hue, hue)
            sample_frame_list = [F.adjust_brightness(x, bright_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_contrast(x, contrast_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_saturation(x, sat_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_hue(x, hue_f) for x in sample_frame_list]
        sample_frame_list = [np.asarray(x, np.float32) for x in sample_frame_list]
        sample_frame_list = [resize(x, self.image_size, interpolation=cv2.INTER_AREA) for x in sample_frame_list]
        sample_frame_list = [x - self.mean for x in sample_frame_list]
        sample_frame_list = [np.transpose(x, (2, 0, 1)) for x in sample_frame_list]
        sample_frame_list_npy = np.stack(sample_frame_list, axis=1)
        # change to float32
        sample_frame_list_npy = np.array(sample_frame_list_npy/255.0, dtype=np.float32)
        return sample_frame_list_npy, action_name, video_name


class NATOPS_select(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=False, use_crop=True, sampling="very_random"):
        super(NATOPS_select, self).__init__()
        self.sampling = sampling
        self.mean = mean
        self.is_jitter = color_jitter

        self.use_crop = use_crop
        if self.use_crop:
            self.y_min, self.y_max, self.x_min, self.x_max = 10, 239, 30, 290
            print("use crop box:", self.y_min, self.y_max, self.x_min, self.x_max)

        self.action_list = ["I Have Command",
                            "All Clear",
                            "Not Clear",
                            "Spread Wings",
                            "Fold Wings",
                            "Lock Wings",
                            "Up Hook",
                            "Down Hook",
                            "Remove Tiedowns",
                            "Remove Chocks",
                            "Insert Chocks",
                            "Move Ahead",
                            "Turn Left",
                            "Turn Right",
                            "Next Marshaller",
                            "Slow Down",
                            "Stop",
                            "Nosegear Steering",
                            "Hot Brakes",
                            "Brakes On",
                            "Brakes Off",
                            "Install Tiedowns",
                            "Fire",
                            "Cut Engine"]
        self.num_frames = num_frames
        self.image_size = image_size
        video_name_list = os.listdir(data_dir)
        video_name_list.sort()

        train_ID = [3, 4, 8, 9, 12, 13, 15, 17, 19, 20]
        test_ID = [2, 5, 6, 7, 10, 11, 14, 16, 18]

        self.ID = train_ID + test_ID

        test_video_name_list = []
        for video_name in video_name_list:
            sub_name = int(video_name[4:6])
            if sub_name in self.ID:
                test_video_name_list.append(video_name)

        self.num_combs = len(self.ID) * len(self.action_list)

        self.video_path_list = [os.path.join(data_dir, video_name) for video_name in test_video_name_list]

        # group each video according to subject and expression
        self.video_dict = {}
        self.num_subjects = len(self.ID)
        for comb_idx in range(self.num_combs):
            sub_idx = comb_idx % self.num_subjects
            action_idx = comb_idx // self.num_subjects
            sub_name = self.ID[sub_idx]
            action_name = self.action_list[action_idx]
            if sub_name not in self.video_dict.keys():
                self.video_dict[sub_name] = {}
            self.video_dict[sub_name][action_name] = []

        for video_path in self.video_path_list:
            video_name = os.path.basename(video_path)
            action_idx = int(video_name[1:3])
            action_name = self.action_list[action_idx - 1]
            sub_name = int(video_name[4:6])
            assert sub_name in self.ID
            assert action_name in self.action_list
            self.video_dict[sub_name][action_name].append(video_path)

    def __len__(self):
        return self.num_combs

    def select(self, sub_name, action_name):
        video_path_list = self.video_dict[sub_name][action_name]
        assert len(video_path_list) > 0
        video_path = str(np.random.choice(video_path_list, size=1)[0])
        video_name = os.path.basename(video_path)
        frame_name_list = os.listdir(video_path)
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        total_num_frames = len(frame_path_list)
        if total_num_frames >= self.num_frames:
            # uniform sampling
            sample_idx_list = np.linspace(start=0, stop=total_num_frames-1, num=self.num_frames, dtype=int)
        else:
            # simply repeat the final frame
            sample_idx_list = np.pad(list(range(total_num_frames)), (0, self.num_frames-total_num_frames), "edge")

        # very random sampling
        if self.sampling == "very_random":
            sample_idx_list = np.sort(np.random.choice(total_num_frames, self.num_frames, replace=True))
            # make the first frame to be 0
            sample_idx_list[0] = 0

        sample_frame_path_list = [frame_path_list[x] for x in sample_idx_list]
        # read image
        sample_frame_list = [imageio.v2.imread(x) for x in sample_frame_path_list]
        # crop image
        if self.use_crop:
            sample_frame_list = [x[self.y_min:self.y_max, self.x_min:self.x_max, :] for x in sample_frame_list]
        # data augmentation
        sample_frame_list = [Image.fromarray(x) for x in sample_frame_list]
        if self.is_jitter:
            bright = 64. / 255
            contrast = 0.25
            sat = 0.25
            hue = 0.04
            bright_f = random.uniform(max(0, 1 - bright), 1 + bright)
            contrast_f = random.uniform(max(0, 1 - contrast), 1 + contrast)
            sat_f = random.uniform(max(0, 1 - sat), 1 + sat)
            hue_f = random.uniform(-hue, hue)
            sample_frame_list = [F.adjust_brightness(x, bright_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_contrast(x, contrast_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_saturation(x, sat_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_hue(x, hue_f) for x in sample_frame_list]
        sample_frame_list = [np.asarray(x, np.float32) for x in sample_frame_list]
        sample_frame_list = [resize(x, self.image_size, interpolation=cv2.INTER_AREA) for x in sample_frame_list]
        sample_frame_list = [x - self.mean for x in sample_frame_list]
        sample_frame_list = [np.transpose(x, (2, 0, 1)) for x in sample_frame_list]
        sample_frame_list_npy = np.stack(sample_frame_list, axis=1)
        # change to float32
        sample_frame_list_npy = np.array(sample_frame_list_npy/255.0, dtype=np.float32)
        return sample_frame_list_npy, action_name, video_name

    def __getitem__(self, index):
        sub_idx = index % self.num_subjects
        action_idx = index // self.num_subjects
        sub_name = self.ID[sub_idx]
        action_name = self.action_list[action_idx]
        video_path_list = self.video_dict[sub_name][action_name]
        assert len(video_path_list) > 0
        video_path = str(np.random.choice(video_path_list, size=1)[0])
        video_name = os.path.basename(video_path)
        frame_name_list = os.listdir(video_path)
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        total_num_frames = len(frame_path_list)
        if total_num_frames >= self.num_frames:
            # uniform sampling
            sample_idx_list = np.linspace(start=0, stop=total_num_frames-1, num=self.num_frames, dtype=int)
        else:
            # simply repeat the final frame
            sample_idx_list = np.pad(list(range(total_num_frames)), (0, self.num_frames-total_num_frames), "edge")

        # very random sampling
        if self.sampling == "very_random":
            sample_idx_list = np.sort(np.random.choice(total_num_frames, self.num_frames, replace=True))
            # make the first frame to be 0
            sample_idx_list[0] = 0

        sample_frame_path_list = [frame_path_list[x] for x in sample_idx_list]
        # read image
        sample_frame_list = [imageio.v2.imread(x) for x in sample_frame_path_list]
        # crop image
        if self.use_crop:
            sample_frame_list = [x[self.y_min:self.y_max, self.x_min:self.x_max, :] for x in sample_frame_list]
        # data augmentation
        sample_frame_list = [Image.fromarray(x) for x in sample_frame_list]
        if self.is_jitter:
            bright = 64. / 255
            contrast = 0.25
            sat = 0.25
            hue = 0.04
            bright_f = random.uniform(max(0, 1 - bright), 1 + bright)
            contrast_f = random.uniform(max(0, 1 - contrast), 1 + contrast)
            sat_f = random.uniform(max(0, 1 - sat), 1 + sat)
            hue_f = random.uniform(-hue, hue)
            sample_frame_list = [F.adjust_brightness(x, bright_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_contrast(x, contrast_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_saturation(x, sat_f) for x in sample_frame_list]
            sample_frame_list = [F.adjust_hue(x, hue_f) for x in sample_frame_list]
        sample_frame_list = [np.asarray(x, np.float32) for x in sample_frame_list]
        sample_frame_list = [resize(x, self.image_size, interpolation=cv2.INTER_AREA) for x in sample_frame_list]
        sample_frame_list = [x - self.mean for x in sample_frame_list]
        sample_frame_list = [np.transpose(x, (2, 0, 1)) for x in sample_frame_list]
        sample_frame_list_npy = np.stack(sample_frame_list, axis=1)
        # change to float32
        sample_frame_list_npy = np.array(sample_frame_list_npy/255.0, dtype=np.float32)
        return sample_frame_list_npy, action_name, video_name


if __name__ == "__main__":
    data_dir = "/data/hfn5052/text2motion/dataset/NATOPS/split_img_data"
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = NATOPS_gen(data_dir=data_dir)
    print(len(dataset))
    train_dataset = data.DataLoader(dataset=dataset,
                                    batch_size=10,
                                    num_workers=4,
                                    shuffle=False)
    for i, batch in enumerate(train_dataset):
        print(i)
