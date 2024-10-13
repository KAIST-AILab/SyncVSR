import os

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule

from datamodule.av_dataset import AVDataset
from .transforms import VideoTransform
from glob import glob

def pad(samples, pad_val=0.0):
    """ https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517 """
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths

def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = torch.cuda.device_count()
        
        self.train_filenames = glob(f"/data/{self.cfg.dataset}/train/*/*.pkl")
        self.val_filenames = glob(f"/data/{self.cfg.dataset}/val/*/*.pkl")
        self.test_filenames = glob(f"/data/{self.cfg.dataset}/test/*/*.pkl")


    def _dataloader(self, ds, collate_fn):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            # batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        train_ds = AVDataset(
            self.train_filenames,
            modality=self.cfg.data.modality,
            audio_transform=None,
            video_transform=VideoTransform("train"),
            language=self.cfg.data.language
        )
        return self._dataloader(train_ds, collate_pad)

    def val_dataloader(self):
        val_ds = AVDataset(
            self.val_filenames,
            modality=self.cfg.data.modality,
            audio_transform=None,
            video_transform=None,
            language=self.cfg.data.language
        )
        return self._dataloader(val_ds, collate_pad)

    def test_dataloader(self):
        dataset = AVDataset(
            self.test_filenames,
            modality=self.cfg.data.modality,
            audio_transform=None,
            video_transform=None,
            language=self.cfg.data.language
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
