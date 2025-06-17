#!/bin/bash

ENV_NAME="graphon"
PYTHON_VERSION="3.9"

echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION and RDKit..."
conda create -y -n $ENV_NAME -c conda-forge python=$PYTHON_VERSION rdkit=2023.03.2

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing graph-tool..."
conda install -y -c conda-forge graph-tool=2.45

echo "Installing PyTorch 2.0.1 with CUDA 11.8 (compatible with driver 12.6)..."
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setting up LD_PRELOAD fix for graph-tool..."
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export LD_PRELOAD=\$(dirname \$(which gcc))/../lib/libgomp.so.1" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo "Setup complete"