#!/bin/bash
# startup.sh
#lsof -ti :1234 | xargs kill -9 

# Verifica che lo script sia eseguito come root
if [ "$EUID" -ne 0 ]; then
    echo "Per favore, esegui questo script come root."
    exit 1
fi

echo "Aggiornamento dei pacchetti e installazione di wget, curl e gnupg2..."
apt-get update && apt-get install -y wget curl gnupg2 lsof
apt-get update && apt-get install -y libgl1 libglib2.0-0

apt update && apt install curl -y
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
mv cloudflared /usr/local/bin/

echo "Installazione di Node.js versione 20 e npm..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

echo "Installazione di Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install

echo "Installazione di Miniconda in /root/miniconda3..."
mkdir -p /root/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh
bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3
rm /root/miniconda3/miniconda.sh

# Aggiorna il PATH per rendere disponibili i comandi conda
export PATH="/root/miniconda3/bin:$PATH"

echo "Inizializzazione di conda e creazione dell'ambiente 'whatifmirror' con Python 3.10..."
source /root/miniconda3/bin/activate
conda init --all
conda create -n whatifmirror python=3.10 -y

echo "Attivazione dell'ambiente 'whatifmirror' e aggiornamento di pip..."
source /root/miniconda3/bin/activate whatifmirror

pip install --upgrade pip
echo "Installazione dei pacchetti pip..."

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "Installazione dei pacchetti pip dal file requirements.txt..."

python3 -m pip install --extra-index-url https://pypi.ngc.nvidia.com \
  onnx-graphsurgeon==0.5.8 polygraphy==0.49.14

pip install -r /StreamDiffusion/requirements.txt

python setup.py StreamDiffusion

echo "Pulizia della cache pip..."
pip cache purge
apt clean

echo "Aggiungo 'conda activate whatifmirror' al file /root/.bashrc per attivare l'ambiente automaticamente nelle nuove sessioni..."
echo "conda activate whatifmirror" >> /root/.bashrc

chmod +x /workspace/StreamDiffusion/mirror/start.sh

echo "Setup completato. Avvio di una shell interattiva nell'ambiente 'whatifmirror'..."
exec bash --login