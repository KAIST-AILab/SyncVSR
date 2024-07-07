from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from transform import create_transform


@dataclass
class LRWDataset(Dataset):
    filenames: list[str]
    categories: list[str]
    transform: Callable[[np.ndarray], np.ndarray]
    durations: pd.DataFrame = None

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        landmarks = np.load(self.filenames[index])
        frame_length = landmarks.shape[0]
        landmarks = torch.as_tensor(landmarks, dtype=torch.float32)

        landmarks = self.transform(landmarks=landmarks)["landmarks"]
        landmarks = landmarks.nan_to_num(0).flatten(1)

        label = self.categories.index(self.filenames[index].split("/")[-3])
        label = torch.tensor(label, dtype=torch.int64)

        audio_filename = self.filenames[index].replace("/mnt/disks/vsr", "/mnt/disks/vsr/data/audio-tokens").replace("npy", "pkl")
        audio_tensor = torch.load(audio_filename)["vq_tokens"].squeeze(0) # or gumbel_audio_tokens

        if self.durations is None:
            return landmarks, label, audio_tensor
        
        example_name = "/".join(self.filenames[index].split("/")[-2:])[:-4]
        word_boundary = int(self.durations.loc[example_name].length)
        start_index = (frame_length - word_boundary) // 2
        end_index = start_index + word_boundary
        word_mask = torch.zeros(frame_length, dtype=torch.long)
        word_mask[start_index:end_index] = 1.0

        return landmarks, label, audio_tensor, word_mask

def create_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    categories = sorted(os.listdir(args.dataset_path))
    if args.use_word_boundary:
        durations = pd.read_csv(args.duration_path, index_col="id")

    train_filenames = glob.glob(os.path.join(args.dataset_path, "*/train/*.npy"))
    valid_filenames = glob.glob(os.path.join(args.dataset_path, "*/val/*.npy"))
    test_filenames = glob.glob(os.path.join(args.dataset_path, "*/test/*.npy"))

    train_transform = create_transform("finetune", max_length=args.input_length)
    valid_transform = create_transform("none", max_length=args.input_length)

    if args.use_word_boundary:
        train_dataset = LRWDataset(train_filenames, categories, train_transform, durations)
        valid_dataset = LRWDataset(valid_filenames, categories, valid_transform, durations)
        test_dataset = LRWDataset(test_filenames, categories, valid_transform, durations)
    else:
        train_dataset = LRWDataset(train_filenames, categories, train_transform)
        valid_dataset = LRWDataset(valid_filenames, categories, valid_transform)
        test_dataset = LRWDataset(test_filenames, categories, valid_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_train_workers,
        persistent_workers=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=args.num_valid_workers,
        persistent_workers=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.valid_batch_size,
        num_workers=args.num_valid_workers,
        persistent_workers=True,
        drop_last=True,
    )
    return train_dataloader, valid_dataloader, test_dataloader
