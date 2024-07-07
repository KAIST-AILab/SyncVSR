import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple

from fairseq.models.wav2vec.wav2vec import Wav2VecModel

__all__ = [
    "CutMix",
    "Compose",
    "Normalize",
    "CenterCrop",
    "RgbToGray",
    "RandomCrop",
    "HorizontalFlip",
    "AddNoise",
    "NormalizeUtterance",
    "TimeMask",
]


class CutMix(nn.Module):
    """
    CutMix : https://arxiv.org/pdf/1905.04899.pdf
    """

    def __init__(
        self,
        num_labels: int,
        wav2vec: Wav2VecModel,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.wav2vec = wav2vec

    def forward(
        self,
        videos: torch.Tensor,
        audios: torch.Tensor,
        labels: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(videos)
        target_ids = torch.randint(0, batch_size, (batch_size,))
        target_rates = torch.rand(batch_size)

        audio_tokens = self.wav2vec.feature_extractor(audios)
        audio_tokens = self.wav2vec.vector_quantizer.forward_idx(audio_tokens)[1]

        mixed_videos, mixed_audios, mixed_labels, mixed_word_masks = [], [], [], []

        for i in range(batch_size):
            org_video, org_audio, org_label, org_word_mask = (
                videos[i],
                audio_tokens[i],
                labels[i],
                word_mask[i],
            )
            tar_video, tar_audio, tar_label, tar_word_mask = (
                videos[target_ids[i]],
                audio_tokens[target_ids[i]],
                labels[target_ids[i]],
                word_mask[target_ids[i]],
            )

            org_pair = (org_video, org_audio, org_label, org_word_mask)
            tar_pair = (tar_video, tar_audio, tar_label, tar_word_mask)

            mixed_video, mixed_audio, mixed_label, mixed_word_mask = self.mask_video(
                org_pair, tar_pair, target_rates[i].item()
            )

            # collect into list
            mixed_videos.append(mixed_video.unsqueeze(0))
            mixed_audios.append(mixed_audio.unsqueeze(0))
            mixed_labels.append(mixed_label.unsqueeze(0))
            mixed_word_masks.append(mixed_word_mask.unsqueeze(0))

        # stack into batch
        video_tensor = torch.cat(mixed_videos, dim=0)
        audio_tensor = torch.cat(mixed_audios, dim=0)
        label_tensor = torch.cat(mixed_labels, dim=0)
        word_mask_tensor = torch.cat(mixed_word_masks, dim=0)

        return video_tensor, audio_tensor, label_tensor, word_mask_tensor

    def mask_video(
        self,
        org_pair: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tar_pair: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mix_rate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cut_out_flag = torch.randint(0, 2, (1,))[0].item()

        org_video, org_audio, org_label, org_word_mask = org_pair
        tar_video, tar_audio, tar_label, tar_word_mask = tar_pair

        org_label_one_hot = F.one_hot(org_label, self.num_labels)
        tar_label_one_hot = F.one_hot(tar_label, self.num_labels)

        if cut_out_flag == 1:
            frame_length = org_video.shape[1]
            cut_off_length = int(frame_length * mix_rate)

            if cut_off_length > 0:
                cut_off_start = torch.randint(
                    0, frame_length - cut_off_length, (1,)
                ).item()

                org_video[0][
                    cut_off_start : cut_off_start + cut_off_length, :, :
                ] = tar_video[0][cut_off_start : cut_off_start + cut_off_length, :, :]
                org_audio[
                    cut_off_start : cut_off_start + cut_off_length, :
                ] = tar_audio[cut_off_start : cut_off_start + cut_off_length, :]

                org_label_one_hot = (
                    1.0 - mix_rate
                ) * org_label_one_hot + mix_rate * tar_label_one_hot
                org_word_mask = (
                    1.0 - mix_rate
                ) * org_word_mask + mix_rate * tar_word_mask

        return org_video, org_audio, org_label_one_hot, org_word_mask


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RgbToGray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CenterCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.0)
        delta_h = int(round((h - th)) / 2.0)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally."""

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


class NormalizeUtterance:
    """Normalize per raw audio by removing the mean and divided by the standard deviation"""

    def __call__(self, signal):
        signal_std = 0.0 if np.std(signal) == 0.0 else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std


class AddNoise(object):
    """Add SNR noise [-1, 1]"""

    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [
            np.float32,
            np.float64,
        ], "noise only supports float data type"

        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2**2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [
            np.float32,
            np.float64,
        ], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise) - len(signal))
            noise_clip = self.noise[start_idx : start_idx + len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power) / (10 ** (snr_target / 10.0))
            desired_signal = (signal + noise_clip * np.sqrt(factor)).astype(np.float32)
            return desired_signal


class TimeMask:
    """time mask"""

    def __init__(self, T=6400, n_mask=2, replace_with_zero=False, inplace=False):
        self.n_mask = n_mask
        self.T = T

        self.replace_with_zero = replace_with_zero
        self.inplace = inplace

    def __call__(self, x):
        if self.inplace:
            cloned = x
        else:
            cloned = x.copy()

        len_raw = cloned.shape[0]
        ts = np.random.randint(0, self.T, size=(self.n_mask, 2))
        for t, mask_end in ts:
            if len_raw - t <= 0:
                continue
            t_zero = random.randrange(0, len_raw - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            if self.replace_with_zero:
                cloned[t_zero:mask_end] = 0
            else:
                cloned[t_zero:mask_end] = cloned.mean()
        return cloned
