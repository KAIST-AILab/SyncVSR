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
conda create -n lrw-tpu python=3.9.13 -y
eval "$(conda shell.bash hook)"
conda activate lrw-tpu
pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U flax optax chex webdataset timm wandb

pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
pip install transformers scikit-learn wandb