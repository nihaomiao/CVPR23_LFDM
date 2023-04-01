# preprocess NATOPS dataset

import os
import cv2
import re
import json_tricks as json
import numpy as np
import imageio
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def save_seg_dict():
    json_path = "/data/hfn5052/text2motion/dataset/NATOPS/data/segmentation.json"
    seg_txt_path = "/data/hfn5052/text2motion/dataset/NATOPS/data/segmentation.txt"
    with open(seg_txt_path, "r") as f:
        lines = f.read().splitlines()
    # deal with segmentation text
    start_line_list = []
    for idx, line in enumerate(lines):
        if line.startswith("// Subject"):
            start_line_list.append(idx)

    pattern = r'\d*,\d*,\d*'
    pattern2 = r'\d*:\d*,\d*'
    split_pattern2 = r'[:|,|]'
    split_dict = {}
    for subject_idx in range(1, 21):
        if subject_idx < 20:
            subject_lines = lines[start_line_list[subject_idx-1]: start_line_list[subject_idx]]
        else:
            subject_lines = lines[start_line_list[subject_idx-1]:]
        subject_start_line_list = []
        for idx, line in enumerate(subject_lines):
            if re.match(pattern, line):
                subject_start_line_list.append(idx)

        split_dict[subject_idx] = {}
        for action_idx in range(1, 25):
            if action_idx < 24:
                action_lines = subject_lines[subject_start_line_list[action_idx-1]: subject_start_line_list[action_idx]]
            else:
                action_lines = subject_lines[subject_start_line_list[action_idx-1]:]
            assert re.match(pattern, action_lines[0])

            split_dict[subject_idx][action_idx] = []
            for idx, line in enumerate(action_lines[1:]):
                if re.match(pattern2, line):
                    result2 = re.split(split_pattern2, line)
                    split_dict[subject_idx][action_idx].append((int(result2[1]), int(result2[2])))

    with open(json_path, "w") as f:
        json.dump(split_dict, f)


def save_split_video():
    # save cropped image and gif
    data_dir = "/data/hfn5052/text2motion/dataset/NATOPS/data/"
    split_dir = "/data/hfn5052/text2motion/dataset/NATOPS/split_img_data"
    os.makedirs(split_dir, exist_ok=True)

    json_path = "/data/hfn5052/text2motion/dataset/NATOPS/data/segmentation.json"
    with open(json_path, "r") as f:
        split_dict = json.load(f)

    for action_idx in range(1, 25):
        action_name = "gesture%02d" % action_idx
        for subject_idx in range(1, 21):
            rgb_video_name = "g%02ds%02d.avi" % (action_idx, subject_idx)
            print(rgb_video_name)
            # read videos
            rgb_video_path = os.path.join(data_dir, action_name, rgb_video_name)
            frame_list = []
            cap = cv2.VideoCapture(rgb_video_path)
            assert cap.get(cv2.CAP_PROP_FPS) == 20
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_list.append(frame[:, :, ::-1])
                else:
                    break
            cap.release()

            cur_split = split_dict[str(subject_idx)][str(action_idx)]
            assert len(cur_split) == 20
            for split_idx in tqdm(range(1, 21)):
                start_frame_idx = cur_split[split_idx - 1][0] - 1
                end_frame_idx = cur_split[split_idx - 1][1] - 1
                split_video_name = "g%02ds%02dp%02d" % (action_idx, subject_idx, split_idx)
                split_video_path = os.path.join(split_dir, split_video_name)
                os.makedirs(split_video_path, exist_ok=True)
                cnt = 0
                for frame_idx in range(start_frame_idx, end_frame_idx + 1):
                    frame_name = split_video_name + "-%04d-%04d.png" % (cnt, frame_idx)
                    imageio.imsave(os.path.join(split_video_path, frame_name),
                                   frame_list[frame_idx])
                    cnt += 1


def split_train_test():
    # random split train/testing
    subject_list = list(range(1, 21))
    random.seed(3407)
    random.shuffle(subject_list)
    train_list = subject_list[:10]
    train_list.sort()
    test_list = subject_list[10:]
    test_list.sort()
    print(train_list)
    # remove 1
    # [3, 4, 8, 9, 12, 13, 15, 17, 19, 20]
    print(test_list)
    # [2, 5, 6, 7, 10, 11, 14, 16, 18]


def anaylze_natops():
    data_dir = "/data/hfn5052/text2motion/dataset/NATOPS/split_img_data"
    video_name_list = os.listdir(data_dir)
    num_frame_list = []
    for video_name in video_name_list:
        video_path = os.path.join(data_dir, video_name)
        frame_list = os.listdir(video_path)
        num_frame = len(frame_list)
        num_frame_list.append(num_frame)
    num_frame_list = np.array(num_frame_list)
    print(num_frame_list.min(), num_frame_list.max(),  num_frame_list.mean())


if __name__ == "__main__":
    # split_train_test()
    # save_seg_dict()
    # save_split_video()
    anaylze_natops()

