# Deep Variational Information Bottleneck -> Under Construction  🏗️


# Installation 
```bash 
conda create -n vib python=3.10 -y
conda activate vib
pip install -e . 
```

If you want to install torch with a specific cuda version in mind, please use the correct torch and cuda wheel. For example:
```bash 
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

To use the VAE libary as a module in this project, you must add it to the python path. Go to the root of this 
```bash 
export PYTHONPATH=/path/to/src/VAEs/:$PYTHONPATH
```
