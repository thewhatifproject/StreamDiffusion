#!/bin/bash
# startup.sh

set -euo pipefail

ENV_NAME="streamdiffusion"        # <-- ambiente conda dedicato
REPO_URL="https://github.com/thewhatifproject/StreamDiffusion.git"
REPO_DIR="/StreamDiffusion"
MINICONDA_DIR="/root/miniconda3"
LOG_FILE="/var/log/streamdiffusion_start.log"

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

echo "Clono il repository in $REPO_DIR ..."
cd /
if [ -d "$REPO_DIR/.git" ]; then
    echo "$REPO_DIR esiste già, eseguo pull..."
    cd "$REPO_DIR"
    git pull --rebase
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
git lfs pull || true

echo "Installazione di Miniconda in $MINICONDA_DIR ..."
mkdir -p "$MINICONDA_DIR"
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_DIR/miniconda.sh"
bash "$MINICONDA_DIR/miniconda.sh" -b -u -p "$MINICONDA_DIR"
rm "$MINICONDA_DIR/miniconda.sh"

# Rende disponibili i comandi conda
export PATH="$MINICONDA_DIR/bin:$PATH"

echo "Inizializzazione di conda per le shell..."
source "$MINICONDA_DIR/bin/activate"
conda init --all || true

echo "Config base conda (show_channel_urls, priority strict)..."
conda config --system --set show_channel_urls true
conda config --system --set channel_priority strict
conda config --system --set auto_update_conda false

echo "Aggiorno 'conda' nel base usando SOLO conda-forge (niente ToS defaults)..."
conda install -n base -y conda -c conda-forge --override-channels || true

echo "Provo ad accettare i Terms of Service dei canali Anaconda..."
tos_supported=0
if conda help tos >/dev/null 2>&1; then
  tos_supported=1
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true
else
  echo "[AVVISO] La versione di conda in uso non fornisce 'conda tos'."
fi

# Se non posso accettare i ToS, uso SOLO conda-forge per evitare repo.anaconda.com
if [ "$tos_supported" -eq 0 ]; then
  echo "Switch ai canali conda-forge (evito repo.anaconda.com)..."
  conda config --system --remove-key channels || true
  conda config --system --add channels conda-forge
  conda config --system --set channel_priority strict
fi

echo "Creazione dell'ambiente '$ENV_NAME' con Python 3.10..."
if [ "$tos_supported" -eq 1 ]; then
  conda create -n "$ENV_NAME" python=3.10 -y
else
  conda create -n "$ENV_NAME" python=3.10 -y -c conda-forge --override-channels
fi

echo "Attivo '$ENV_NAME' e aggiorno pip..."
source "$MINICONDA_DIR/bin/activate" "$ENV_NAME"
python -m pip install --upgrade pip

# --- UNINSTALL PREVENTIVO: rimuovi eventuale onnxruntime (CPU) ---
pip uninstall -y onnxruntime onnxruntime-silicon || true

echo "Installo pacchetti pip base (Torch CUDA 12.8, ONNX tools)..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install --extra-index-url https://pypi.ngc.nvidia.com \
  onnx-graphsurgeon==0.5.8 polygraphy==0.49.14

# --- Installa toolchain ONNX/ORT corretta (GPU, IR>=11) ---
python -m pip install --upgrade --force-reinstall \
  onnx==1.18.0 onnxruntime-gpu==1.22.0 "protobuf>=3.20.2,<5" "cuda-python>=12.8,<12.9"

echo "Installazione requirements dal repository clonato..."
python -m pip install -r "$REPO_DIR/requirements.txt"

echo "Installazione del pacchetto StreamDiffusion dal sorgente (editable, extras all)..."
cd "$REPO_DIR"
python -m pip install -e .[all]

# --- UNINSTALL PREVENTIVO (bis): se il setup avesse rimesso ORT CPU, rimuovilo e ripristina GPU ---
pip uninstall -y onnxruntime || true
python -m pip install --upgrade --force-reinstall onnxruntime-gpu==1.22.0

# --- Sanity check: ORT deve supportare IR>=11 ---
python - <<'PY'
import onnx, onnxruntime as ort, sys
print("onnx:", onnx.__version__)
print("onnxruntime:", ort.__version__)
from onnx import helper
m = helper.make_model(helper.make_graph([], "g", [], []))
m.ir_version = 11
try:
    ort.InferenceSession(m.SerializeToString())
    print("OK: onnxruntime supporta IR>=11")
except Exception as e:
    print("ERRORE ORT:", e)
    sys.exit(1)
PY

echo "Imposto permessi ed eseguo lo start.sh in altra shell..."
chmod +x "$REPO_DIR/mirror/start.sh" || true
nohup bash -lc "source $MINICONDA_DIR/bin/activate $ENV_NAME && $REPO_DIR/mirror/start.sh" \
  >"$LOG_FILE" 2>&1 & disown || true
echo "Log: $LOG_FILE"

echo "Pulizia cache pip e apt..."
pip cache purge || true
apt-get clean

echo "Abilito l'attivazione automatica dell'ambiente conda '$ENV_NAME' nelle nuove sessioni..."
grep -qxF "conda activate $ENV_NAME" /root/.bashrc || echo "conda activate $ENV_NAME" >> /root/.bashrc

echo "Setup completato. Avvio di una shell interattiva nell'ambiente '$ENV_NAME'..."
exec bash --login