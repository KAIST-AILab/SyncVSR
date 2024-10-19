# encoding: utf-8
import os
import math
import glob
import pandas as pd
from typing import Union, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from tqdm import tqdm
import mediapipe as mp
from pydub import AudioSegment


target_dir = "./LRW/lipread_dataset"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

jpeg = TurboJPEG()
mp_face_mesh = mp.solutions.face_mesh


FACEMESH_LIPS = [
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
]


def retrieve_landmark(path):
    # replace .mp4 with .npy
    path = path.replace(".mp4", ".npy")
    # read landmark
    return np.load(path)


def retrieve_txt(path):
    # replace .mp4 with .txt
    path = path.replace(".mp4", ".txt")
    # read text
    with open(path, "r") as f:
        text = f.read()
    return text


def _normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int = 256,
    image_height: int = 256,
) -> Union[None, Tuple[int, int]]:
    """
    Converts normalized value pair to pixel coordinates.
    https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
    """

    x_px = math.floor(normalized_x * image_width)
    y_px = math.floor(normalized_y * image_height)
    return x_px, y_px


def extract_opencv(filename):
    landmark = retrieve_landmark(filename)
    list_bbox = []

    for frame_idx in range(landmark.shape[0]):
        # print(landmark)
        x_coords, y_coords, z_coords = (
            landmark[frame_idx][:, 0],
            landmark[frame_idx][:, 1],
            landmark[frame_idx][:, 2],
        )
        # get max of x, y, z
        min_x, min_y, min_z = np.min(x_coords), np.min(y_coords), np.min(z_coords)
        max_x, max_y, max_z = np.max(x_coords), np.max(y_coords), np.max(z_coords)

        # extract FACEMESH_LIPS
        x_coords = x_coords[FACEMESH_LIPS]
        y_coords = y_coords[FACEMESH_LIPS]
        # print(x_coords.shape, y_coords.shape)

        median_x = (np.median(x_coords) + np.mean(x_coords) + min_x + max_x) / 4
        median_y = (np.median(y_coords) + np.mean(y_coords) + min_y + max_y) / 4
        median_x, median_y = _normalized_to_pixel_coordinates(median_x, median_y,)
        bounding_box = [
            median_x - 48 - 8,
            median_y - 48 - 8,
            median_x + 48 + 8,
            median_y + 48 + 8,
        ]
        list_bbox.append(bounding_box)

    cap = cv2.VideoCapture(filename)
    frame_idx = 0
    video = []
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            lookup_idx = frame_idx
            left_x, top_y, right_x, bottom_y = list_bbox[lookup_idx]

            if left_x <= 0 or top_y <= 0 or right_x >= 256 or bottom_y >= 256:
                print("out of bound for file ", filename, " at frame ", frame_idx)
                x_target_size = 112.0
                y_target_size = 112.0
                if left_x <= 0.0:
                    left_x = 0.0
                    right_x = left_x + x_target_size
                if top_y <= 0.0:
                    top_y = 0.0
                    bottom_y = top_y + y_target_size
                if right_x >= 256.0:
                    right_x = 256.0
                    left_x = right_x - x_target_size
                if bottom_y >= 256.0:
                    bottom_y = 256.0
                    top_y = bottom_y - y_target_size

            cropped_frame = frame[
                int(top_y) : int(bottom_y), int(left_x) : int(right_x)
            ]
            if cropped_frame.shape != (112, 112, 3):
                raise Exception(
                    "Error in frame", filename, "with shape of", cropped_frame.shape
                )
            cropped_frame = jpeg.encode(cropped_frame)
            video.append(cropped_frame)
            frame_idx += 1
        else:
            break

    cap.release()
    return video


class LRWDataset(Dataset):
    def __init__(self):
        root_dir = "./LRW/lipread_mp4/"

        with open("./LRW/labels.txt") as myfile:
            labels = myfile.read().splitlines()

        all_files = []
        for label in tqdm(labels):
            word_label_files = glob.glob(os.path.join(root_dir, label, "*", "*.mp4"))
            word_label_files.sort()
            all_files.extend(word_label_files)

        print(f"Total number of files: {len(all_files)} with {len(labels)} labels")

        self.all_files = all_files

    def __getitem__(self, idx):
        file_name = self.all_files[idx]
        dir_name = os.path.dirname(file_name)
        base_name = os.path.basename(file_name)

        result = {}

        result["video"] = extract_opencv(file_name)
        result["audio"] = AudioSegment.from_file(file_name, format="mp4")
        result["text"] = retrieve_txt(file_name)

        savename = file_name.replace("dset_mp4", "lipread_dataset").replace(
            ".mp4", ".pkl"
        )
        # if the folder does not exist, create it
        if not os.path.exists(os.path.dirname(savename)):
            os.makedirs(os.path.dirname(savename))
        torch.save(result, savename)

        return result

    def __len__(self):
        return len(self.all_files)


if __name__ == "__main__":
    loader = DataLoader(
        LRWDataset(),
        batch_size=128,
        num_workers=128,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x,
    )

    for i, batch in enumerate(tqdm(loader)):
        pass
