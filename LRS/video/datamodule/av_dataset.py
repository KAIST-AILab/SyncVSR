import random
import numpy as np

import torch
import torchvision
from turbojpeg import TJPF_GRAY, TurboJPEG

from preprocess.utils import pydub_to_np
from .transforms import TextTransform, FunctionalModule


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filenames,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        language=None,
    ):
        self.filenames = filenames
        self.jpeg = TurboJPEG()

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.lrs3_video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.Resize(96),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )
        self.lrs2_video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(96),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

        self.length = np.load("./datamodule/video_length.npy")
        self.length_distribution = np.bincount(self.length)
        self.cut = self.length.max()
        self.tokenizer = TextTransform(
            sp_model_path="./spm/unigram/unigram5000.model",
            dict_path="./spm/unigram/unigram5000_units.txt",
        )
        self.video_max_length = self.length.max()
        self.language = language
        self.probability_distribution = self.length_distribution/self.length_distribution.sum()
        self.neural_audio_codec_sample_rate = 16000
        self.audio_padding_max = int((self.video_max_length / 25) * self.neural_audio_codec_sample_rate)
        
        # multiple for video-audio alignment
        self.audio_multiple = 40
        print(f"using {self.language} with {self.cut} cut")


    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = torch.load(filename)

        # load from dictionary
        video = data["video"]
        video_size = len(video)
        audio = data["audio"]
        text = data["text"]
        text_data = text.split("\n")

        if video_size > self.cut: # pretrain or vox2
            # shard the text
            margin = 0.5 if "LRS3" in filename else 0.2

            # get random starting point and crop video and audio
            random_start = random.randint(0, video_size - self.cut)
            sample_length = np.random.choice(len(self.length_distribution), p=self.probability_distribution)
            video = video[random_start:random_start+sample_length]
            audio = audio[random_start*self.audio_multiple:(random_start+sample_length)*self.audio_multiple]

            # parse text for pretrain files with timestamps
            words = [text_data[i].split(" ") for i in range(4, len(text_data)-1)]
            
            # cut and align video, text and audio
            time_start = random_start / 25
            time_end = time_start + (sample_length / 25)
            
            list_word = []
            for word in words:
                if (time_start - margin) <= float(word[1]) and float(word[2]) < (time_end + margin):
                    list_word.append(word[0])
            
            text_string = " ".join(list_word)
    
        else: # main or short vox2
            text_string = text_data[0][5:].strip()
        
        token_id = self.tokenizer.tokenize(text_string)

        # post processing and transforming into tensor
        video = np.stack([self.jpeg.decode(img, TJPF_GRAY) for img in video])
        video = torch.as_tensor(video).permute(0, 3, 1, 2) # T x C x H x W
        if self.video_transform:
            video = self.video_transform(video)
        else:
            if "LRS3" in filename:
                video = self.lrs3_video_pipeline(video)
            elif "LRS2" in filename:
                video = self.lrs2_video_pipeline(video)

        # load audio
        audio, sampling_rate = pydub_to_np(audio)
        audio = torch.as_tensor(audio)
        audio = audio.squeeze()

        if len(audio) < len(video) * self.audio_multiple:
            print(filename, len(audio), len(video))

        return {"input": video, "audio": audio, "target": token_id}

    def __len__(self):
        return len(self.filenames)
