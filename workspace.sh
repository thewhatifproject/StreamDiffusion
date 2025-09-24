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
git lfs pull || true

echo "Installazione di Miniconda in /root/miniconda3..."
mkdir -p /root/miniconda3
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh
bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3
rm /root/miniconda3/miniconda.sh

# Aggiorna il PATH per rendere disponibili i comandi conda
export PATH="/root/miniconda3/bin:$PATH"

echo "Inizializzazione di conda (shells)..."
source /root/miniconda3/bin/activate
conda init --all || true

echo "Config base conda (show_channel_urls, priority strict)..."
conda config --system --set show_channel_urls true
conda config --system --set channel_priority strict
conda config --system --set auto_update_conda false

# --- Gestione ToS / Canali ---
echo "Accettazione ToS se supportata, altrimenti switch a conda-forge..."
tos_supported=0
if conda help tos >/dev/null 2>&1; then
  tos_supported=1
  # Prova ad accettare i ToS per i canali Anaconda
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true
else
  echo "[AVVISO] La versione di conda non supporta 'conda tos'. Userò solo conda-forge."
fi

# Se non supporta tos (o vuoi evitare repo.anaconda.com), forza canali = solo conda-forge
if [ "$tos_supported" -eq 0 ]; then
  # rimuovi TUTTI i canali e aggiungi solo conda-forge
  conda config --system --remove-key channels || true
  conda config --system --add channels conda-forge
  conda config --system --set channel_priority strict
fi

echo "Creazione dell'ambiente 'whatifmirror' con Python 3.10..."
if [ "$tos_supported" -eq 1 ]; then
  # usa i canali configurati (defaults/forge a seconda di come sei messo)
  conda create -n whatifmirror python=3.10 -y
else
  # forza SOLO conda-forge per evitare repo.anaconda.com
  conda create -n whatifmirror python=3.10 -y -c conda-forge --override-channels
fi

echo "Attivazione dell'ambiente 'whatifmirror' e aggiornamento di pip..."
source /root/miniconda3/bin/activate whatifmirror
python -m pip install --upgrade pip

echo "Installazione dei pacchetti pip di base (Torch CUDA 12.8, ONNX tools)..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -m pip install --extra-index-url https://pypi.ngc.nvidia.com \
  onnx-graphsurgeon==0.5.8 polygraphy==0.49.14

echo "Installazione requirements dal repository clonato..."
python -m pip install -r /StreamDiffusion/requirements.txt

echo "Installazione del pacchetto StreamDiffusion dal sorgente..."
cd /StreamDiffusion
# Se setup.py non serve, puoi sostituire con: python -m pip install -e .[all]
python setup.py develop easy_install streamdiffusion[all] || true

echo "Imposto permessi di esecuzione sugli script del progetto..."
chmod +x /StreamDiffusion/mirror/start.sh || true

echo "Pulizia della cache pip e apt..."
pip cache purge || true
apt-get clean

echo "Abilito l'attivazione automatica dell'ambiente conda nelle nuove sessioni..."
grep -qxF 'conda activate whatifmirror' /root/.bashrc || echo "conda activate whatifmirror" >> /root/.bashrc

echo "Avvio dello start.sh in un'altra shell (sessione separata)...
- log: /var/log/streamdiffusion_start.log"
# Avvia start.sh in una nuova shell di login con l'ambiente conda attivo, in background
nohup bash -lc 'source /root/miniconda3/bin/activate whatifmirror && /StreamDiffusion/mirror/start.sh' \
  >/var/log/streamdiffusion_start.log 2>&1 & disown || true

echo "Setup completato. Avvio di una shell interattiva nell'ambiente 'whatifmirror'..."
exec bash --login