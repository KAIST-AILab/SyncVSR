# encoding: utf-8
import os
import glob
from tqdm import tqdm
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from pydub import AudioSegment

from utils import retrieve_txt

# get number of cpus
jpeg = TurboJPEG()

def centercrop_opencv(filename):
    left_x, top_y, right_x, bottom_y = [48, 48, 176, 176]

    cap = cv2.VideoCapture(filename)
    video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cropped_frame = frame[
                int(top_y) : int(bottom_y), int(left_x) : int(right_x)
            ]
            cropped_frame = jpeg.encode(cropped_frame)
            video.append(cropped_frame)
        else:
            break
    cap.release()
    return video


class VoxDataset(Dataset):
    def __init__(self, root_dir="/data/vox2", target_dir="/data/vox2-pkl"):
        self.root_dir = root_dir
        self.target_dir = target_dir

        all_txt_files = glob.glob(os.path.join(self.root_dir, "**", "mp4", "**", "**", "*.txt"))
        all_txt_files.sort()
        all_mp4_files = [filename.replace(".txt", ".mp4") for filename in all_txt_files]
        self.all_files = all_mp4_files

    def __getitem__(self, idx):
        file_name = self.all_files[idx]

        result = {}
        result["video"] = centercrop_opencv(file_name)
        result["audio"] = AudioSegment.from_file(
            file_name, format="mp4"
        )
        result["text"] = retrieve_txt(file_name)

        savename = file_name.replace(self.root_dir, self.target_dir).replace(".mp4", ".pkl")
        # if the folder does not exist, create it
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        torch.save(result, savename)


    def __len__(self):
        return len(self.all_files)


if __name__ == "__main__":
    num_cpus = os.cpu_count() - 1
    print("Number of cpus: {}".format(num_cpus))

    target_dir = "/data/vox2-pkl"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    loader = DataLoader(
        VoxDataset(
            root_dir="/data/vox2",
            target_dir=target_dir,
        ),
        batch_size=num_cpus,
        num_workers=num_cpus,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x,
    )

    for i, batch in enumerate(tqdm(loader)):
        pass