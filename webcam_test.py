import cv2
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from PIL import Image

# Load the local model
pipe = StableDiffusionPipeline.from_single_file("../models/checkpoints/kohaku-v2.1.safetensors").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

# Initialize StreamDiffusion
stream = StreamDiffusion(
    pipe,
    t_index_list=[32, 45],
    torch_dtype=torch.float16,
)

# Load and fuse LCM
stream.load_lcm_lora()
stream.fuse_lora()

# Enable acceleration
pipe.enable_xformers_memory_efficient_attention()

# Setup webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Test generation
prompt = "1girl with dog hair, thick frame glasses"
stream.prepare(prompt)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert BGR to RGB and resize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb).resize((512, 512))
    
    # Generate image
    x_output = stream(frame_pil)
    output_image = postprocess_image(x_output, output_type="pil")[0]
    
    # Convert back to BGR for display
    output_cv = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)
    
    # Display
    cv2.imshow('StreamDiffusion', output_cv)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 