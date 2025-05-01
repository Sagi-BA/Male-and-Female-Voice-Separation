#!/bin/bash

# Update system packages
apt-get update
apt-get upgrade -y

# Install system dependencies
apt-get install -y ffmpeg
apt-get install -y libsndfile1
apt-get install -y python3-pip
apt-get install -y python3-venv

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0 