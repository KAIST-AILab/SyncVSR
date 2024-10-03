# SyncVSR

### [SyncVSR: Data-Efficient Visual Speech Recognition with End-to-End Crossmodal Audio Token Synchronization (Interspeech 2024)](https://arxiv.org/abs/2406.12233)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/lipreading-on-lip-reading-in-the-wild)](https://paperswithcode.com/sota/lipreading-on-lip-reading-in-the-wild?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/lipreading-on-lrw-1000)](https://paperswithcode.com/sota/lipreading-on-lrw-1000?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/lipreading-on-lrs2)](https://paperswithcode.com/sota/lipreading-on-lrs2?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/lipreading-on-lrs3-ted)](https://paperswithcode.com/sota/lipreading-on-lrs3-ted?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/landmark-based-lipreading-on-lrw)](https://paperswithcode.com/sota/landmark-based-lipreading-on-lrw?p=syncvsr-data-efficient-visual-speech)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/syncvsr-data-efficient-visual-speech/landmark-based-lipreading-on-lrs2)](https://paperswithcode.com/sota/landmark-based-lipreading-on-lrs2?p=syncvsr-data-efficient-visual-speech)  


Visual Speech Recognition (VSR) stands at the intersection of computer vision and speech recognition, aiming to interpret spoken content from visual cues. A prominent challenge in VSR is the presence of homophenes-visually similar lip gestures that represent different phonemes. Prior approaches have sought to distinguish fine-grained visemes by aligning visual and auditory semantics, but often fell short of full synchronization. To address this, we present SyncVSR, an end-to-end learning framework that leverages quantized audio for frame-level crossmodal supervision. By integrating a projection layer that synchronizes visual representation with acoustic data, our encoder learns to generate discrete audio tokens from a video sequence in a non-autoregressive manner. SyncVSR shows versatility across tasks, languages, and modalities at the cost of a forward pass. Our empirical evaluations show that it not only achieves state-of-the-art results but also reduces data usage by up to ninefold.

### Overview of SyncVSR

**Frame-level crossmodal supervision with quantized audio tokens for enhanced Visual Speech Recognition.**

|                     Overview of SyncVSR                      |                Performance of SyncVSR on LRS3                |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img width="300" alt="image" src="./others/SyncVSR.png?raw=true"> | <img width="440" alt="image" src="./others/LRS3.png?raw=true"> |



```python3
class Model(nn.Module):
    """
    - audio_alignment: Ratio of audio tokens per video frame
    - vq_groups: Number of quantized audio groups (i.e. audio channels number in the output of the codec)
    - audio_vocab_size: Vocabulary size of quantized audio tokens of neural audio codec
    - audio_projection: Linear projection layer for audio reconstruction
    """
    def __init__(self, config):
        ...
        self.audio_projection = nn.Linear(config.hidden_size, audio_alignment * vq_groups * audio_vocab_size)
        self.lambda_audio = 10.0 # Larger the better, recommending at least 10 times larger loss coefficient of the VSR objective

    def forward(self, videos, audio_tokens, ...):
        # Get traditional VSR objective loss such as Word classification loss, CTC loss, and LM loss
        loss_objective = ...

        # get latent of the encoder from input video frames
        last_hidden_state = self.encoder(videos) # [B, seq_len+1, hidden_size]

        # Get audio reconstruction loss
        logits_audio = self.audio_projection(last_hidden_state[:, 1:, :]) # [B, seq_len, audio_alignment * vq_groups * audio_vocab_size]
        logits_audio = logits_audio.reshape(B, seq_len, audio_alignment * vq_groups, audio_vocab_size) # [B, seq_len, audio_alignment * vq_groups, audio_vocab_size]
        # For each encoded video frame, it should predict combination of (audio_alignment * vq_groups) audio tokens
        loss_audio = F.cross_entropy(
            logits_audio.reshape(-1, self.audio_vocab_size), # [B * seq_len * (audio_alignment * vq_groups), audio_vocab_size]
            audio_tokens.flatten(), # [B * seq_len * (audio_alignment * vq_groups),]
        )

        # Simply add audio reconstruction loss to the objective loss. That's it!
        loss_total = loss_objective + loss_audio * self.lambda_audio
        ...
```

### Audio Tokens Preparation

**We uploaded tokenized audio for LRW, LRS2, LRS3 at the [release section](https://github.com/KAIST-AILab/SyncVSR/releases/).** Without installing the fairseq environment, you may load the tokenized audio from the files as below:

```bash
# download from the release section below
# https://github.com/KAIST-AILab/SyncVSR/releases/

# and untar the folder.
tar -xf audio-tokens.tar.gz
```

```python3
""" access to the tokenized audio files """
import os
from glob import glob

benchname = "LRW" # or LRS2, LRS3
split = "train"
dataset_path = os.path.join("./data/audio-tokens", benchname)
audio_files = glob(os.path.join(dataset_path, "**", split, "*.pkl"))

""" load the dataset """
import random
import torch

tokenized_audio_sample = torch.load(random.choice(audio_files)) # dictionary type
tokenized_audio_sample.keys() # 'vq_tokens', 'wav2vec2_tokens'
```

### Video Dataset Preparation

1. Get authentification for Lip Reading in the Wild Dataset via https://www.bbc.co.uk/rd/projects/lip-reading-datasets
2. Download dataset using the shell command below

```shell
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaa
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partab
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partac
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partad
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partae
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaf
wget --user <USERNAME> --password <PASSWORD> https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partag
```
3. Extract region of interest and convert mp4 file into pkl file with the commands below.
```shell
python ./src/preprocess_roi.py
python ./src/preprocess_pkl.py
```


### Installation
For the replicating state-of-the-art results from the scratch, please follow the instructions below.
```shell
# install depedency through apt-get
apt-get update 
apt-get -yq install ffmpeg libsm6 libxext6 
apt install libturbojpeg tmux -y

# create conda virtual env
wget https://repo.continuum.io/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh -b
source /root/anaconda3/bin/activate base
conda create -n lip python=3.9.13 -y
source /root/anaconda3/bin/activate lip

# install dependencies
git clone https://github.com/KAIST-AILab/SyncVSR.git
cd SyncVSR
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
pip install -r requirements.txt
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt -P ./
```

### Train

For training with our methodology, please run the command below after preprocessing the dataset. You may change configurations in yaml files.
```shell
python ./src/train.py ./config/bert-12l-512d.yaml devices=[0] # Transformer backbone
```

### Inference

For inference, please download the pretrained checkpoint from the repository's [release section](https://github.com/KAIST-AILab/SyncVSR/releases/) and run the code with the following command.
```shell
python ./src/inference.py ./config/bert-12l-512d.yaml devices=[0] # Transformer backbone
```
