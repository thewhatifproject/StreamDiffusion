#!/bin/bash
lsof -ti :1234 | xargs kill -9 

# Vai nella cartella del frontend, installa le dipendenze e costruisci il progetto
cd frontend
npm install
npm run build

# Torna alla cartella principale, poi passa a quella del backend
cd ..

# Avvia il backend in background
python3 main.py --port 1234 --host 0.0.0.0 &
BACKEND_PID=$!

# For ControlNet mode, add --controlnet-config parameter:
# python3 main.py --port 7860 --host 0.0.0.0 --controlnet-config ../../configs/controlnet_examples/depth_example.yaml 

# Attendi qualche secondo che il server parta
sleep 5

# Avvia il tunnel Cloudflare (senza dominio, usa trycloudflare.com)
cloudflared tunnel --url http://localhost:1234