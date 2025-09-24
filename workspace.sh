#!/bin/bash
# startup.sh

set -euo pipefail

# lsof -ti :1234 | xargs -r kill -9

# Verifica che lo script sia eseguito come root
if [ "$EUID" -ne 0 ]; then
    echo "Per favore, esegui questo script come root."
    exit 1
fi

echo "Aggiornamento dei pacchetti e installazione di utility di base..."
apt-get update && apt-get install -y wget curl gnupg2 lsof git

echo "Installazione di librerie runtime necessarie..."
apt-get install -y libgl1 libglib2.0-0

echo "Installazione cloudflared..."
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
mv cloudflared /usr/local/bin/

echo "Installazione di Node.js 20 e npm..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

echo "Installazione di Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install

echo "Clono il repository in /StreamDiffusion..."
# Clona nella cartella principale dell'ambiente "/"
cd /
if [ -d "/StreamDiffusion/.git" ]; then
    echo "/StreamDiffusion esiste già, eseguo pull..."
    cd /StreamDiffusion
    git pull --rebase
else
    git clone https://github.com/thewhatifproject/StreamDiffusion.git /StreamDiffusion
    cd /StreamDiffusion
fi
# Recupero eventuali asset LFS
git lfs pull

echo "Installazione di Miniconda in /root/miniconda3..."
mkdir -p /root/miniconda3
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh
bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3
rm /root/miniconda3/miniconda.sh

# Aggiorna il PATH per rendere disponibili i comandi conda
export PATH="/root/miniconda3/bin:$PATH"

echo "Inizializzazione di conda e creazione dell'ambiente 'whatifmirror' con Python 3.10..."
source /root/miniconda3/bin/activate
conda init --all
conda create -n whatifmirror python=3.10 -y

echo "Attivazione dell'ambiente 'whatifmirror' e aggiornamento di pip..."
# Attiva l'ambiente
source /root/miniconda3/bin/activate whatifmirror

python -m pip install --upgrade pip
echo "Installazione dei pacchetti pip di base (Torch CUDA 12.8, ONNX tools)..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -m pip install --extra-index-url https://pypi.ngc.nvidia.com \
  onnx-graphsurgeon==0.5.8 polygraphy==0.49.14

echo "Installazione requirements dal repository clonato..."
# requirements del progetto
python -m pip install -r /StreamDiffusion/requirements.txt

echo "Installazione del pacchetto StreamDiffusion dal sorgente..."
cd /StreamDiffusion
python setup.py develop easy_install streamdiffusion[all]

echo "Imposto permessi di esecuzione sugli script del progetto..."
chmod +x /StreamDiffusion/mirror/start.sh || true

echo "Pulizia della cache pip e apt..."
pip cache purge || true
apt-get clean

echo "Abilito l'attivazione automatica dell'ambiente conda nelle nuove sessioni..."
grep -qxF 'conda activate whatifmirror' /root/.bashrc || echo "conda activate whatifmirror" >> /root/.bashrc

echo "Setup completato. Avvio di una shell interattiva nell'ambiente 'whatifmirror'..."
exec bash --login