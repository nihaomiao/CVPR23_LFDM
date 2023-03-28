# Preprocess MHAD dataset to generate cropped video-text pairs
# Using depth map to generate bounding box for cropping videos
import os
import scipy.io
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio


def find_overall_bbox():
    root_dir_path = "/zdata/projects/text2motion/datasets/UTD-MHAD"
    depth_dir_path = os.path.join(root_dir_path, "Depth")

    data_sum = np.zeros((240, 320), dtype=np.int64)
    for idx, action in enumerate(range(1, 28)):
        print(idx)
        for subject in range(1, 9):
            for trial in range(1, 5):
                data = import_depth_data(depth_dir_path, action, subject, trial)
                if data is not None:
                    data_sum += np.sum(data, axis=2, dtype=np.int64)
    # finding bounding box
    data_sum_nz = np.nonzero(data_sum)
    y_min, y_max = data_sum_nz[0].min(), data_sum_nz[0].max()
    x_min, x_max = data_sum_nz[1].min(), data_sum_nz[1].max()
    # y_min: 0, y_max: 239, x_min: 77, x_max: 246


def import_depth_data(depth_dir_path, action, subject, trial):
    filename = os.path.join(depth_dir_path, f'a{action}_s{subject}_t{trial}_depth.mat')
    if Path(filename).is_file():
        mat = scipy.io.loadmat(filename)
        return mat['d_depth']
    else:
        return None


def import_rgb_data(rgb_dir_path, action, subject, trial):
    filename = os.path.join(rgb_dir_path, f'a{action}_s{subject}_t{trial}_color.avi')
    frame_list = []
    if Path(filename).is_file():
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_list.append(frame[:, :, ::-1])
            else:
                break
        return frame_list
    else:
        return None


def crop_rgb_data(rgb_dir_path, action, subject, trial, x_min, x_max, y_min, y_max):
    filename = os.path.join(rgb_dir_path, f'a{action}_s{subject}_t{trial}_color.avi')
    frame_list = []
    if Path(filename).is_file():
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_list.append(frame[y_min:y_max, x_min:x_max, ::-1])
            else:
                break
        return frame_list
    else:
        return None


def crop_video():
    root_dir_path = "/zdata/projects/text2motion/datasets/UTD-MHAD"
    video_dir_path = os.path.join(root_dir_path, "RGB")
    depth_dir_path = os.path.join(root_dir_path, "Depth")
    save_dir_path = os.path.join(root_dir_path, "crop_rgb")
    os.makedirs(save_dir_path, exist_ok=True)

    RGB_H = 480
    RGB_W = 640
    Y_min = 0
    Y_max = 239 + 1
    X_min = 78
    X_max = 245 + 1
    Y_min *= 2
    Y_max *= 2
    X_min *= 2
    X_max *= 2
    Y_min = max(0, Y_min)
    Y_max = min(RGB_H, Y_max)
    X_min = max(0, X_min)
    X_max = min(RGB_W, X_max)

    for idx, action in enumerate(range(1, 28)):
        print(idx)
        for subject in range(1, 9):
            for trial in range(1, 5):
                depth_data = import_depth_data(depth_dir_path, action, subject, trial)
                if depth_data is None:
                    continue
                frame_list = import_rgb_data(video_dir_path, action, subject, trial)
                # crop video
                crop_frame_list = [frame[Y_min:Y_max, X_min:X_max, :] for frame in frame_list]
                # print(crop_frame_list[0].shape)
                # (480, 336)
                filename = f'a{action}_s{subject}_t{trial}_crop.mp4'
                imageio.mimsave(os.path.join(save_dir_path, filename), crop_frame_list)


def save_crop_image():
    root_dir_path = "/data/hfn5052/text2motion/MHAD"
    video_dir_path = os.path.join(root_dir_path, "RGB")
    depth_dir_path = os.path.join(root_dir_path, "Depth")
    save_dir_path = os.path.join(root_dir_path, "crop_image")
    os.makedirs(save_dir_path, exist_ok=True)

    RGB_H = 480
    RGB_W = 640
    Y_min = 0
    Y_max = 239 + 1
    X_min = 78
    X_max = 245 + 1
    Y_min *= 2
    Y_max *= 2
    X_min *= 2
    X_max *= 2
    Y_min = max(0, Y_min)
    Y_max = min(RGB_H, Y_max)
    X_min = max(0, X_min)
    X_max = min(RGB_W, X_max)

    for idx, action in enumerate(range(1, 28)):
        print(idx)
        for subject in range(1, 9):
            for trial in range(1, 5):
                depth_data = import_depth_data(depth_dir_path, action, subject, trial)
                if depth_data is None:
                    continue
                frame_list = import_rgb_data(video_dir_path, action, subject, trial)
                # crop video
                crop_frame_list = [frame[Y_min:Y_max, X_min:X_max, :] for frame in frame_list]
                image_dir_name = f'a{action}_s{subject}_t{trial}'
                image_dir_path = os.path.join(save_dir_path, image_dir_name)
                os.makedirs(image_dir_path, exist_ok=True)
                for image_idx, crop_frame in enumerate(crop_frame_list):
                    crop_frame_name = image_dir_name + "_%03d.png" % image_idx
                    crop_frame_path = os.path.join(image_dir_path, crop_frame_name)
                    imageio.imsave(crop_frame_path, crop_frame)


def analyse_MHAD():
    data_dir = "/data/hfn5052/text2motion/dataset/MHAD/crop_image"
    video_name_list = os.listdir(data_dir)
    video_name_list.sort()
    video_path_list = [os.path.join(data_dir, x) for x in video_name_list]
    min_num_frame = 1e4
    max_num_frame = -1
    min_video_name = None
    max_video_name = None
    num_frame_list = []
    for video_path in video_path_list:
        frame_name_list = os.listdir(video_path)
        num_frame = len(frame_name_list)
        if num_frame < min_num_frame:
            min_num_frame = num_frame
            min_video_name = video_path
        if num_frame > max_num_frame:
            max_num_frame = num_frame
            max_video_name = video_path
        num_frame_list.append(num_frame)
    num_frame_list = np.array(num_frame_list)
    print(min_num_frame, min_video_name)
    print(max_num_frame, max_video_name)
    print(num_frame_list.min(), num_frame_list.max(), num_frame_list.mean())
    # 32 /data/hfn5052/text2motion/MHAD/crop_image/a15_s2_t4
    # 96 /data/hfn5052/text2motion/MHAD/crop_image/a21_s8_t1


def split_train_test_MHAD():
    male = [1, 5, 6, 8]
    female = [2, 3, 4, 7]
    train_sub = [1, 5, 2, 3]
    test_sub = [6, 8, 4, 7]


if __name__ == "__main__":
    # the train/test split can be found in:
    # split_train_test_MHAD()

    # use this function to perform the cropping
    # save_crop_image()
    pass
