from __future__ import annotations

import glob
import os
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import DictConfig
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from turbojpeg import TJPF_GRAY, TurboJPEG
from augment import *


@dataclass
class LRWDatasetForDCTCN(Dataset):
    filenames: list[str]
    labels: list[str]
    transform: nn.Module
    durations: pd.DataFrame
    max_time_masks: int | None = None
    validation: bool = False
    jpeg: TurboJPEG = TurboJPEG()

    def __len__(self) -> int:
        return len(self.filenames)

    def _mask_frames(self, batch: dict[str, torch.Tensor]):
        length = np.random.randint(self.max_time_masks)
        offset = np.random.randint(batch["videos"].size(0) - length)
        batch["videos"][offset : offset + length] = batch["videos"].mean()

    def _trim_frames(self, batch: dict[str, torch.Tensor]):
        audio_stride = batch["videos"].size(0) / batch["audios"].size(0)
        boundary_size = batch["word_mask"].sum()
        truncated_length = np.random.randint(boundary_size, batch["videos"].size(0))

        offset = np.random.randint(truncated_length - boundary_size + 1)
        shift = int(offset - (batch["videos"].size(0) - boundary_size) // 2)

        batch["videos"] = batch["videos"].roll(shift, 0)
        batch["videos"][truncated_length:] = 0

        batch["audios"] = batch["audios"].roll(int(shift / audio_stride), 0)
        batch["audios"][int(truncated_length / audio_stride) :] = 0

        batch["word_mask"] = batch["word_mask"].roll(shift, 0)
        batch["word_mask"][truncated_length:] = 0

        batch["attention_mask"] = batch["attention_mask"].roll(shift, 0)
        batch["attention_mask"][truncated_length:] = 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = torch.load(self.filenames[index])
        batch = {"audios": data["audio"]}

        example_name = "/".join(self.filenames[index].split("/")[-2:])[:-4]
        word_boundary = int(self.durations.loc[example_name].length)

        label = self.filenames[index].split("/")[-3]
        batch["labels"] = torch.tensor(self.labels.index(label), dtype=torch.int64)

        video = np.stack([self.jpeg.decode(img, TJPF_GRAY) for img in data["video"]])
        video = torch.as_tensor(video).permute(0, 3, 1, 2)
        batch["videos"] = 2 * video.float() / 0xFF - 1

        start_index = (video.size(0) - word_boundary) // 2
        end_index = start_index + word_boundary

        batch["attention_mask"] = torch.ones(video.size(0), dtype=torch.long)
        batch["word_mask"] = torch.zeros(video.size(0), dtype=torch.long)
        batch["word_mask"][start_index:end_index] = 1

        if not self.validation:
            self._mask_frames(batch)
            self._trim_frames(batch)

        video = batch["videos"]
        video = video.squeeze(1)
        video = self.transform(video)

        batch["videos"] = video.unsqueeze(1)
        batch["videos"] = batch["videos"].permute(1, 0, 2, 3)
        return batch


@dataclass
class LRWDatasetForTransformer(Dataset):
    filenames: list[str]
    labels: list[str]
    transform: nn.Module | None = None
    durations: pd.DataFrame = None
    jpeg: TurboJPEG = TurboJPEG()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = torch.load(self.filenames[index])
        label = self.filenames[index].split("/")[-3]
        label = torch.tensor(self.labels.index(label), dtype=torch.int64)

        video = [self.jpeg.decode(img, pixel_format=TJPF_GRAY) for img in data["video"]]
        frame_length = len(video)
        video = torch.as_tensor(np.stack(video)).permute(0, 3, 1, 2)
        # video = 2 * video.float() / 0xFF - 1

        video = video.permute(1, 0, 2, 3)
        if self.transform is not None:
            video_array = video[0].numpy()
            video_array = self.transform(video_array)
            video = torch.tensor(video_array, dtype=torch.float).unsqueeze(0)

        # Word Boundary Information
        example_name = "/".join(self.filenames[index].split("/")[-2:])[:-4]
        word_boundary = int(self.durations.loc[example_name].length)
        start_index = (frame_length - word_boundary) // 2
        end_index = start_index + word_boundary
        word_mask = torch.zeros(frame_length, dtype=torch.long)
        word_mask[start_index:end_index] = 1.0

        return video, data["audio"], label, word_mask


def create_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader]:

    labels = sorted(os.listdir(config.data.label_directory))
    durations = pd.read_csv(config.data.durations, index_col="id")

    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)

    if config.model.name == "transformer" :

        train_transforms = Compose(
            [
                Normalize(0.0, 255.0),
                RandomCrop(crop_size),
                HorizontalFlip(0.5),
                Normalize(mean, std),
                TimeMask(T=0.6 * 25, n_mask=1),
            ]
        )
        val_test_transforms = Compose(
            [Normalize(0.0, 255.0), CenterCrop(crop_size), Normalize(mean, std)]
        )
    else :
        train_transforms = [
            T.RandomCrop(config.data.input_size),
            T.RandomHorizontalFlip(),
        ]
        train_transforms = T.Compose(train_transforms)
        val_test_transform = T.CenterCrop(config.data.input_size)

    if config.model.name == "transformer":
        train_dataset = LRWDatasetForTransformer(
            glob.glob(config.data.train), labels, train_transforms, durations
        )
        validation_dataset = LRWDatasetForTransformer(
            glob.glob(config.data.validation), labels, val_test_transforms, durations
        )
        test_dataset = LRWDatasetForTransformer(
            glob.glob(config.data.test), labels, val_test_transforms, durations
        )
    elif config.model.name == "dc-tcn":
        train_dataset = LRWDatasetForDCTCN(
            glob.glob(config.data.train),
            labels,
            train_transforms,
            durations,
            config.data.max_time_masks,
            validation=False,
        )
        validation_dataset = LRWDatasetForDCTCN(
            glob.glob(config.data.validation),
            labels,
            val_test_transforms,
            durations,
            config.data.max_time_masks,
            validation=True,
        )
        test_dataset = LRWDatasetForDCTCN(
            glob.glob(config.data.test),
            labels,
            val_test_transforms,
            durations,
            config.data.max_time_masks,
            validation=True,
        )
    else:
        raise NotImplementedError("Not valid model name")

    num_workers = multiprocessing.cpu_count() // 4

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.train.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader, test_dataloader
