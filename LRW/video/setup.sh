#!/bin/bash

# Check if conda is already installed
if ! command -v conda &> /dev/null
then
    echo "Miniconda is not installed. Proceeding with installation..."
    
    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
    
    # Run the Miniconda installer
    bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -b -u
    
    # Remove the installer
    rm Miniconda3-py310_24.1.2-0-Linux-x86_64.sh

    # Initialize conda
    ~/miniconda3/bin/conda init bash
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
    
    echo "Miniconda installation and setup complete."
else
    echo "Miniconda is already installed."
fi

# 2. Install requirements.
conda create -n lrw python=3.9.13 -y
eval "$(conda shell.bash hook)"
conda activate lrw
pip install pytorch-lightning==1.9.1 x-transformers==1.9.2
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install pandas==1.5.3 numpy==1.26.0 

pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
pip install omegaconf sentencepiece transformers scikit-learn pydub timm wandb
pip install opencv-python mediapipe
wget https://github.com/KAIST-AILab/SyncVSR/releases/download/weight-audio-v1/LRW_CKPT_epoch_167_step_213864.ckpt