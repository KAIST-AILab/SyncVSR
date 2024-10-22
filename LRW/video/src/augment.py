import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple
from utils import check_availability

if check_availability("fairseq"):
    from fairseq.models.wav2vec.wav2vec import Wav2VecModel


class CutMix(nn.Module):
    """
    CutMix: https://arxiv.org/pdf/1905.04899.pdf
    Sequential frame cutmix
    """

    def __init__(
        self,
        num_labels: int,
        wav2vec,
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

        if self.wav2vec:
            audio_tokens = self.wav2vec.feature_extractor(audios)
            audio_tokens = self.wav2vec.vector_quantizer.forward_idx(audio_tokens)[1]
        else:
            audio_tokens = audios

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

class TimeMask(nn.Module):
    def __init__(self, T=6400, n_mask=2, replace_with_zero=False):
        super(TimeMask, self).__init__()
        self.T = T
        self.n_mask = n_mask
        self.replace_with_zero = replace_with_zero

    def forward(self, x):
        cloned = x.clone()
        len_raw = cloned.size(0)

        for _ in range(self.n_mask):
            t = random.randint(0, min(self.T, len_raw))
            t_zero = random.randint(0, len_raw - t)
            mask_end = t_zero + t
            
            if self.replace_with_zero:
                cloned[t_zero:mask_end] = 0
            else:
                cloned[t_zero:mask_end] = cloned.mean()
                
        return cloned
    
class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)
