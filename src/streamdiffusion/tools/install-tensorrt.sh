#!/bin/bash
set -e

# Aggiornamento e installazione di Python e dipendenze per il build
apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Installazione di Torch
pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
pip uninstall -y triton
rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Installazione di TensorRT
pip install --no-cache-dir tensorrt==10.7
rm -rf /root/.cache/pip /tmp/* /var/tmp/*

pip uninstall -y triton

# Preparazione del layer dei pacchetti Python (escludendo torch e nvidia)
export SEPARATE_PACKAGES="torch|nvidia"
mkdir -p /opt/layers/python-packages
find /usr/local/lib/python3.10/dist-packages -mindepth 1 -maxdepth 1 \
    -not -path "/usr/local/lib/python3.10/dist-packages/nvidia*" \
    -not -path "/usr/local/lib/python3.10/dist-packages/torch*" \
    -exec cp -r {} /opt/layers/python-packages/ \;

# Preparazione dei layer Nvidia
export NVIDIA_PACKAGES="cuda cublas cudnn cufft curand cusolver cusparse nccl nvjitlink nvtx"
for pkg in $NVIDIA_PACKAGES; do
    mkdir -p /opt/layers/${pkg}
    mkdir -p /opt/layers/${pkg}/nvidia/${pkg}
    cp -r /usr/local/lib/python3.10/dist-packages/nvidia_${pkg}* /opt/layers/${pkg} 2>/dev/null || true
    cp -r /usr/local/lib/python3.10/dist-packages/nvidia/${pkg}* /opt/layers/${pkg}/nvidia/ 2>/dev/null || true
done

# Preparazione del layer Torch
mkdir -p /opt/layers/torch
cp -r /usr/local/lib/python3.10/dist-packages/torch* /opt/layers/torch/

# Installazione degli strumenti per l'ambiente runtime
apt-get update && \
    apt-get install -y --no-install-recommends wget curl git python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Aggiornamento di pip, setuptools e wheel
pip install --upgrade pip setuptools wheel

# Impostazione della variabile LD_LIBRARY_PATH per CUDA
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

pip uninstall -y triton
rm -rf /root/.cache/pip /tmp/* /var/tmp/*

echo "Installazione completata."