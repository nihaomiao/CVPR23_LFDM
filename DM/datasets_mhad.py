# loading MHAD dataset for training and testing
import os
import random

import numpy as np
import torch.utils.data as data
import imageio
from misc import resize
import cv2
import torchvision.transforms.functional as F
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


class MHAD(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128, transform=None,
                 mean=(0, 0, 0), color_jitter=True, split_train_test=True,
                 sampling="random"):
        super(MHAD, self).__init__()
        self.mean = mean
        self.is_jitter = color_jitter
        self.sampling = sampling

        self.action_list = ["right arm swipe to the left",
                            "right arm swipe to the right",
                            "right hand wave",
                            "two hand front clap",
                            "right arm throw",
                            "cross arms in the chest",
                            "basketball shooting",
                            "draw x",
                            "draw circle clockwise",
                            "draw circle counter clockwise",
                            "draw triangle",
                            "right hand bowling",
                            "front boxing",
                            "baseball swing from right",
                            "tennis forehand swing",
                            "two arms curl",
                            "tennis serve",
                            "two hand push",
                            "knock on door",
                            "hand catch",
                            "pick up and throw",
                            "jogging",
                            "walking",
                            "sit to stand",
                            "stand to sit",
                            "forward lunge (left foot forward)",
                            "squat"]
        self.num_frames = num_frames
        self.image_size = image_size
        video_name_list = os.listdir(data_dir)
        video_name_list.sort()

        if split_train_test:
            train_ID = [1, 5, 2, 3]
            train_video_name_list = []
            for video_name in video_name_list:
                sub_name = int(video_name.split("_")[1][1:])
                if sub_name in train_ID:
                    train_video_name_list.append(video_name)

        if split_train_test:
            self.video_path_list = [os.path.join(data_dir, video_name) for video_name in train_video_name_list]
        else:
            self.video_path_list = [os.path.join(data_dir, video_name) for video_name in video_name_list]

    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        video_name = os.path.basename(video_path)
        action_idx = int(video_name.split("_")[0][1:])
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


class MHAD_test(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=256,
                 mean=(0, 0, 0), color_jitter=False, split_train_test=True):
        super(MHAD_test, self).__init__()
        self.mean = mean
        self.is_jitter = color_jitter

        self.action_list = ["right arm swipe to the left",
                            "right arm swipe to the right",
                            "right hand wave",
                            "two hand front clap",
                            "right arm throw",
                            "cross arms in the chest",
                            "basketball shooting",
                            "draw x",
                            "draw circle clockwise",
                            "draw circle counter clockwise",
                            "draw triangle",
                            "right hand bowling",
                            "front boxing",
                            "baseball swing from right",
                            "tennis forehand swing",
                            "two arms curl",
                            "tennis serve",
                            "two hand push",
                            "knock on door",
                            "hand catch",
                            "pick up and throw",
                            "jogging",
                            "walking",
                            "sit to stand",
                            "stand to sit",
                            "forward lunge (left foot forward)",
                            "squat"]
        self.num_frames = num_frames
        self.image_size = image_size
        video_name_list = os.listdir(data_dir)
        video_name_list.sort()

        if split_train_test:
            test_ID = [6, 8, 4, 7]
            test_video_name_list = []
            for video_name in video_name_list:
                sub_name = int(video_name.split("_")[1][1:])
                if sub_name in test_ID:
                    test_video_name_list.append(video_name)

        if split_train_test:
            self.video_path_list = [os.path.join(data_dir, video_name) for video_name in test_video_name_list]
        else:
            self.video_path_list = [os.path.join(data_dir, video_name) for video_name in video_name_list]

    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        video_name = os.path.basename(video_path)
        action_idx = int(video_name.split("_")[0][1:])
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


# for consistently generating videos
class MHAD_gen(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=False, sampling="very_random"):
        super(MHAD_gen, self).__init__()
        self.sampling = sampling
        self.mean = mean
        self.is_jitter = color_jitter

        self.action_list = ["right arm swipe to the left",
                            "right arm swipe to the right",
                            "right hand wave",
                            "two hand front clap",
                            "right arm throw",
                            "cross arms in the chest",
                            "basketball shooting",
                            "draw x",
                            "draw circle clockwise",
                            "draw circle counter clockwise",
                            "draw triangle",
                            "right hand bowling",
                            "front boxing",
                            "baseball swing from right",
                            "tennis forehand swing",
                            "two arms curl",
                            "tennis serve",
                            "two hand push",
                            "knock on door",
                            "hand catch",
                            "pick up and throw",
                            "jogging",
                            "walking",
                            "sit to stand",
                            "stand to sit",
                            "forward lunge (left foot forward)",
                            "squat"]
        self.num_frames = num_frames
        self.image_size = image_size
        video_name_list = os.listdir(data_dir)
        video_name_list.sort()

        self.test_ID = [6, 8, 4, 7]
        test_video_name_list = []
        for video_name in video_name_list:
            sub_name = int(video_name.split("_")[1][1:])
            if sub_name in self.test_ID:
                test_video_name_list.append(video_name)

        self.num_combs = len(self.test_ID) * len(self.action_list)

        self.video_path_list = [os.path.join(data_dir, video_name) for video_name in test_video_name_list]

        # group each video
        self.video_dict = {}
        for comb_idx in range(self.num_combs):
            sub_idx = comb_idx % 4
            action_idx = comb_idx // 4
            sub_name = self.test_ID[sub_idx]
            action_name = self.action_list[action_idx]
            if sub_name not in self.video_dict.keys():
                self.video_dict[sub_name] = {}
            self.video_dict[sub_name][action_name] = []

        for video_path in self.video_path_list:
            video_name = os.path.basename(video_path)
            action_idx = int(video_name.split("_")[0][1:])
            action_name = self.action_list[action_idx - 1]
            sub_name = int(video_name.split("_")[1][1:])
            assert sub_name in self.test_ID
            assert action_name in self.action_list
            self.video_dict[sub_name][action_name].append(video_path)

    def __len__(self):
        return self.num_combs

    def __getitem__(self, index):
        sub_idx = index % 4
        action_idx = index // 4
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


# select one subject and one action
class MHAD_select(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=False, sampling="very_random"):
        super(MHAD_select, self).__init__()
        self.sampling = sampling
        self.mean = mean
        self.is_jitter = color_jitter

        self.action_list = ["right arm swipe to the left",
                            "right arm swipe to the right",
                            "right hand wave",
                            "two hand front clap",
                            "right arm throw",
                            "cross arms in the chest",
                            "basketball shooting",
                            "draw x",
                            "draw circle clockwise",
                            "draw circle counter clockwise",
                            "draw triangle",
                            "right hand bowling",
                            "front boxing",
                            "baseball swing from right",
                            "tennis forehand swing",
                            "two arms curl",
                            "tennis serve",
                            "two hand push",
                            "knock on door",
                            "hand catch",
                            "pick up and throw",
                            "jogging",
                            "walking",
                            "sit to stand",
                            "stand to sit",
                            "forward lunge (left foot forward)",
                            "squat"]
        self.num_frames = num_frames
        self.image_size = image_size
        video_name_list = os.listdir(data_dir)
        video_name_list.sort()

        self.ID = list(range(1, 9))

        self.num_combs = len(self.ID) * len(self.action_list)

        self.video_path_list = [os.path.join(data_dir, video_name) for video_name in video_name_list]

        # group each video according to subject and expression
        self.video_dict = {}
        for comb_idx in range(self.num_combs):
            sub_idx = comb_idx % 8
            action_idx = comb_idx // 8
            sub_name = self.ID[sub_idx]
            action_name = self.action_list[action_idx]
            if sub_name not in self.video_dict.keys():
                self.video_dict[sub_name] = {}
            self.video_dict[sub_name][action_name] = []

        for video_path in self.video_path_list:
            video_name = os.path.basename(video_path)
            action_idx = int(video_name.split("_")[0][1:])
            action_name = self.action_list[action_idx - 1]
            sub_name = int(video_name.split("_")[1][1:])
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

    def __getitem__(self, index):
        sub_idx = index % 8
        action_idx = index // 8
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


if __name__ == "__main__":
    pass

