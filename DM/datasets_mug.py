import os
import random

import numpy as np
import torch.utils.data as data
import imageio
from misc import resize
import cv2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image


class MUG(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(128, 128, 128), color_jitter=True, sampling="random"):
        super(MUG, self).__init__()
        self.mean = mean
        self.is_jitter = color_jitter
        self.sampling = sampling

        train_ID = ['008', '017', '021', '028', '030', '031', '034', '036', '037',
                    '038', '039', '042', '043', '044', '045', '055', '060', '061',
                    '062', '063', '071', '075', '076', '077', '083', '084']
        session_ID = ["002", "003", "049"]
        self.exp_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.num_frames = num_frames
        self.image_size = image_size
        self.video_path_list = []
        for video_name in train_ID:
            if video_name not in session_ID:
                for exp_name in self.exp_list:
                    cur_video_dir_path = os.path.join(data_dir, video_name, exp_name)
                    if os.path.exists(cur_video_dir_path):
                        cur_video_name_list = os.listdir(cur_video_dir_path)
                        cur_video_name_list.sort()
                        for cur_video_name in cur_video_name_list:
                            cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                            self.video_path_list.append(cur_video_path)
            else:
                for session_name in ["session0", "session0"]:
                    for exp_name in self.exp_list:
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
        exp_name = video_name.split("_")[-2]
        assert exp_name in self.exp_list
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
        frame_name_list.sort()
        frame_path_list = [os.path.join(video_path, frame_name) for frame_name in frame_name_list]
        total_num_frames = len(frame_path_list)
        if total_num_frames >= self.num_frames:
            # uniform sampling
            if self.sampling == "uniform":
                sample_idx_list = np.linspace(start=0, stop=total_num_frames-1, num=self.num_frames, dtype=int)
            # uniform random sampling
            if self.sampling == "random":
                uniform_idx_list = np.linspace(start=0, stop=total_num_frames-1, num=self.num_frames, dtype=int)
                step_list = uniform_idx_list[1:] - uniform_idx_list[0:-1]
                sample_idx_list = uniform_idx_list.copy()
                for ii in range(1, self.num_frames - 1):
                    low = 1-step_list[ii-1]
                    high = +step_list[ii]
                    sample_idx_list[ii] = sample_idx_list[ii] + np.random.randint(low=low, high=high)
                sample_idx_list = np.sort(sample_idx_list)
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
        return sample_frame_list_npy, exp_name, video_name


class MUG_test(data.Dataset):
    def __init__(self, data_dir, num_frames=16, image_size=256,
                 mean=(128, 128, 128), color_jitter=False):
        super(MUG_test, self).__init__()
        self.mean = mean
        self.is_jitter = color_jitter

        test_ID = ['001', '002', '006', '007', '010', '013', '014', '020', '027', '032',
                   '033', '040', '046', '048', '049', '052', '064', '065', '066', '070',
                   '072', '073', '074', '078', '079', '082']
        session_ID = ["002", "003", "049"]
        self.exp_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.num_frames = num_frames
        self.image_size = image_size
        self.video_path_list = []
        for video_name in test_ID:
            if video_name not in session_ID:
                for exp_name in self.exp_list:
                    cur_video_dir_path = os.path.join(data_dir, video_name, exp_name)
                    if os.path.exists(cur_video_dir_path):
                        cur_video_name_list = os.listdir(cur_video_dir_path)
                        cur_video_name_list.sort()
                        for cur_video_name in cur_video_name_list:
                            cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                            self.video_path_list.append(cur_video_path)
            else:
                for session_name in ["session0", "session0"]:
                    for exp_name in self.exp_list:
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
        exp_name = video_name.split("_")[-2]
        assert exp_name in self.exp_list
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
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
        return sample_frame_list_npy, exp_name, video_name


# for consistently generating videos
class MUG_gen(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=False, sampling="very_random"):
        super(MUG_gen, self).__init__()
        self.sampling = sampling
        self.mean = mean
        self.is_jitter = color_jitter
        self.test_ID = ['001', '002', '006', '007', '010', '013', '014', '020', '027', '032',
                        '033', '040', '046', '048', '049', '052', '064', '065', '066', '070',
                        '072', '073', '074', '078', '079', '082']
        session_ID = ["002", "003", "049"]
        self.exp_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.num_combs = len(self.test_ID) * len(self.exp_list)
        self.num_frames = num_frames
        self.image_size = image_size
        self.video_path_list = []
        for video_name in self.test_ID:
            if video_name not in session_ID:
                for exp_name in self.exp_list:
                    cur_video_dir_path = os.path.join(data_dir, video_name, exp_name)
                    if os.path.exists(cur_video_dir_path):
                        cur_video_name_list = os.listdir(cur_video_dir_path)
                        cur_video_name_list.sort()
                        for cur_video_name in cur_video_name_list:
                            cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                            self.video_path_list.append(cur_video_path)
            else:
                for session_name in ["session0", "session0"]:
                    for exp_name in self.exp_list:
                        cur_video_dir_path = os.path.join(data_dir, video_name, session_name, exp_name)
                        if os.path.exists(cur_video_dir_path):
                            cur_video_name_list = os.listdir(cur_video_dir_path)
                            cur_video_name_list.sort()
                            for cur_video_name in cur_video_name_list:
                                cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                                self.video_path_list.append(cur_video_path)
        # group each video according to subject and expression
        self.video_dict = {}
        for comb_idx in range(self.num_combs):
            sub_idx = comb_idx % 26
            exp_idx = comb_idx // 26
            sub_name = self.test_ID[sub_idx]
            exp_name = self.exp_list[exp_idx]
            if sub_name not in self.video_dict.keys():
                self.video_dict[sub_name] = {}
            self.video_dict[sub_name][exp_name] = []

        for video_path in self.video_path_list:
            sub_name = video_path.split("/")[6]
            exp_name = video_path.split("/")[-2]
            assert sub_name in self.test_ID
            assert exp_name in self.exp_list
            self.video_dict[sub_name][exp_name].append(video_path)

    def __len__(self):
        return int(self.num_combs)

    def __getitem__(self, index):
        sub_idx = index % 26
        exp_idx = index // 26
        sub_name = self.test_ID[sub_idx]
        exp_name = self.exp_list[exp_idx]
        video_path_list = self.video_dict[sub_name][exp_name]
        if len(video_path_list) == 0:
            video_path_list = self.video_dict[sub_name]["neutral"]
            assert len(video_path_list) > 0
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = sub_name + "_" + exp_name + "_fake"
        else:
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = "_".join(video_path.split("/")[-3:]) if "session" not in video_path \
                else "_".join(video_path.split("/")[-4:])
        exp_name = video_name.split("_")[-2]
        assert exp_name in self.exp_list
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
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
        return sample_frame_list_npy, exp_name, video_name


# for consistently generating training videos
class MUG_gen_train(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=False, sampling="very_random"):
        super(MUG_gen_train, self).__init__()
        self.sampling = sampling
        self.mean = mean
        self.is_jitter = color_jitter
        self.train_ID = ['008', '017', '021', '028', '030', '031', '034', '036', '037',
                         '038', '039', '042', '043', '044', '045', '055', '060', '061',
                         '062', '063', '071', '075', '076', '077', '083', '084']
        session_ID = ["002", "003", "049"]
        self.exp_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.num_combs = len(self.train_ID) * len(self.exp_list)
        self.num_frames = num_frames
        self.image_size = image_size
        self.video_path_list = []
        for video_name in self.train_ID:
            if video_name not in session_ID:
                for exp_name in self.exp_list:
                    cur_video_dir_path = os.path.join(data_dir, video_name, exp_name)
                    if os.path.exists(cur_video_dir_path):
                        cur_video_name_list = os.listdir(cur_video_dir_path)
                        cur_video_name_list.sort()
                        for cur_video_name in cur_video_name_list:
                            cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                            self.video_path_list.append(cur_video_path)
            else:
                for session_name in ["session0", "session0"]:
                    for exp_name in self.exp_list:
                        cur_video_dir_path = os.path.join(data_dir, video_name, session_name, exp_name)
                        if os.path.exists(cur_video_dir_path):
                            cur_video_name_list = os.listdir(cur_video_dir_path)
                            cur_video_name_list.sort()
                            for cur_video_name in cur_video_name_list:
                                cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                                self.video_path_list.append(cur_video_path)
        # group each video according to subject and expression
        self.video_dict = {}
        for comb_idx in range(self.num_combs):
            sub_idx = comb_idx % 26
            exp_idx = comb_idx // 26
            sub_name = self.train_ID[sub_idx]
            exp_name = self.exp_list[exp_idx]
            if sub_name not in self.video_dict.keys():
                self.video_dict[sub_name] = {}
            self.video_dict[sub_name][exp_name] = []

        for video_path in self.video_path_list:
            sub_name = video_path.split("/")[6]
            exp_name = video_path.split("/")[-2]
            assert sub_name in self.train_ID
            assert exp_name in self.exp_list
            self.video_dict[sub_name][exp_name].append(video_path)

    def __len__(self):
        return int(self.num_combs)

    def __getitem__(self, index):
        sub_idx = index % 26
        exp_idx = index // 26
        sub_name = self.train_ID[sub_idx]
        exp_name = self.exp_list[exp_idx]
        video_path_list = self.video_dict[sub_name][exp_name]
        if len(video_path_list) == 0:
            # sample_frame_list_npy = np.zeros((3, self.num_frames, self.image_size, self.image_size), dtype=np.float32)
            # video_name = sub_name + "_" + exp_name
            # return sample_frame_list_npy, exp_name, video_name
            video_path_list = self.video_dict[sub_name]["anger"]
            assert len(video_path_list) > 0
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = sub_name + "_" + exp_name + "_fake"
        else:
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = "_".join(video_path.split("/")[-3:]) if "session" not in video_path \
                else "_".join(video_path.split("/")[-4:])
        exp_name = video_name.split("_")[-2]
        assert exp_name in self.exp_list
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
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
        return sample_frame_list_npy, exp_name, video_name


# select one subject and one action
class MUG_select(data.Dataset):
    def __init__(self, data_dir, num_frames=40, image_size=128,
                 mean=(0, 0, 0), color_jitter=False, sampling="very_random"):
        super(MUG_select, self).__init__()
        self.sampling = sampling
        self.mean = mean
        self.is_jitter = color_jitter
        train_ID = ['008', '017', '021', '028', '030', '031', '034', '036', '037',
                    '038', '039', '042', '043', '044', '045', '055', '060', '061',
                    '062', '063', '071', '075', '076', '077', '083', '084']
        test_ID = ['001', '002', '006', '007', '010', '013', '014', '020', '027', '032',
                   '033', '040', '046', '048', '049', '052', '064', '065', '066', '070',
                   '072', '073', '074', '078', '079', '082']
        self.ID = train_ID + test_ID
        session_ID = ["002", "003", "049"]
        self.exp_list = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        self.num_combs = len(self.ID) * len(self.exp_list)
        self.num_frames = num_frames
        self.image_size = image_size
        self.video_path_list = []
        for video_name in self.ID:
            if video_name not in session_ID:
                for exp_name in self.exp_list:
                    cur_video_dir_path = os.path.join(data_dir, video_name, exp_name)
                    if os.path.exists(cur_video_dir_path):
                        cur_video_name_list = os.listdir(cur_video_dir_path)
                        cur_video_name_list.sort()
                        for cur_video_name in cur_video_name_list:
                            cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                            self.video_path_list.append(cur_video_path)
            else:
                for session_name in ["session0", "session0"]:
                    for exp_name in self.exp_list:
                        cur_video_dir_path = os.path.join(data_dir, video_name, session_name, exp_name)
                        if os.path.exists(cur_video_dir_path):
                            cur_video_name_list = os.listdir(cur_video_dir_path)
                            cur_video_name_list.sort()
                            for cur_video_name in cur_video_name_list:
                                cur_video_path = os.path.join(cur_video_dir_path, cur_video_name)
                                self.video_path_list.append(cur_video_path)
        # group each video according to subject and expression
        self.video_dict = {}
        self.num_subjects = len(self.ID)
        for comb_idx in range(self.num_combs):
            sub_idx = comb_idx % self.num_subjects
            exp_idx = comb_idx // self.num_subjects
            sub_name = self.ID[sub_idx]
            exp_name = self.exp_list[exp_idx]
            if sub_name not in self.video_dict.keys():
                self.video_dict[sub_name] = {}
            self.video_dict[sub_name][exp_name] = []

        for video_path in self.video_path_list:
            sub_name = video_path.split("/")[6]
            exp_name = video_path.split("/")[-2]
            assert sub_name in self.ID
            assert exp_name in self.exp_list
            self.video_dict[sub_name][exp_name].append(video_path)

    def __len__(self):
        return int(self.num_combs)

    def select(self, sub_name, exp_name):
        video_path_list = self.video_dict[sub_name][exp_name]
        if len(video_path_list) == 0:
            video_path_list = self.video_dict[sub_name]["anger"]
            assert len(video_path_list) > 0
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = sub_name + "_" + exp_name + "_fake"
        else:
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = "_".join(video_path.split("/")[-3:]) if "session" not in video_path \
                else "_".join(video_path.split("/")[-4:])
        exp_name = video_name.split("_")[-2]
        assert exp_name in self.exp_list
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
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
        return sample_frame_list_npy, exp_name, video_name

    def __getitem__(self, index):
        sub_idx = index % self.num_subjects
        exp_idx = index // self.num_subjects
        sub_name = self.ID[sub_idx]
        exp_name = self.exp_list[exp_idx]
        video_path_list = self.video_dict[sub_name][exp_name]
        if len(video_path_list) == 0:
            video_path_list = self.video_dict[sub_name]["anger"]
            assert len(video_path_list) > 0
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = sub_name + "_" + exp_name + "_fake"
        else:
            video_path = str(np.random.choice(video_path_list, size=1)[0])
            video_name = "_".join(video_path.split("/")[-3:]) if "session" not in video_path \
                else "_".join(video_path.split("/")[-4:])
        exp_name = video_name.split("_")[-2]
        assert exp_name in self.exp_list
        frame_name_list = os.listdir(video_path)
        frame_name_list = [x for x in frame_name_list if x.endswith("jpg") or x.endswith("png")]
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
        return sample_frame_list_npy, exp_name, video_name


if __name__ == "__main__":
    data_dir = "/data/hfn5052/text2motion/dataset/MUG"

    dataset = MUG_gen_train(data_dir)

    train_dataset = data.DataLoader(dataset=dataset,
                                    batch_size=10,
                                    num_workers=8,
                                    shuffle=False)
    for i, batch in enumerate(train_dataset):
        print(i)
