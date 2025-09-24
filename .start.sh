#!/bin/bash
lsof -ti :1234 | xargs kill -9 

# Vai nella cartella del frontend, installa le dipendenze e costruisci il progetto
cd /StreamDiffusion/demo/realtime-img2img
cd frontend
npm install
npm run build
# Torna alla cartella principale, poi passa a quella del backend
cd ../backend
# Avvia il backend in background

# --host HOST                    Host address (default: 0.0.0.0)
# --port PORT                    Port number (default: 7860)
# --controlnet-config PATH       Path to ControlNet YAML configuration (optional)
# --acceleration ACCEL           Acceleration type: none, xformers, sfast, tensorrt
# --taesd / --no-taesd          Use Tiny Autoencoder (default: enabled)
# --engine-dir DIR              TensorRT engine directory
# --debug                       Enable debug mode

python main.py --acceleration none &
BACKEND_PID=$!

# Attendi qualche secondo che il server parta
sleep 5

# Avvia il tunnel Cloudflare (senza dominio, usa trycloudflare.com)
cloudflared tunnel --url http://localhost:1234