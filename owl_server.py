import base64
from io import BytesIO
import random
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import AutoPipelineForText2Image

app = FastAPI(title="Owl Character Rewards API")

# Enable CORS for local frontend integration
# Wildcard + credentials is an invalid CORS combination; list origins explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# 1. Device Selection (MPS for Apple Silicon, CUDA for Nvidia, CPU fallback)
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32  # MPS handles float32 reliably for SD-Turbo
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

print(f"Loading SD-Turbo onto device: {device.upper()}...")

# 2. Load SD-Turbo Pipeline
try:
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", 
        torch_dtype=torch_dtype
    )
    pipeline.to(device)
    print("Stable Diffusion Turbo loaded successfully!")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    print("Inference will fall back to CPU or local generation.")
    pipeline = None

# Fallback prompts if Ollama is not running
FALLBACK_OWL_TEMPLATES = {
    "styles": ["low-res pixel art", "cute 3D claymation", "retro 8-bit game sprite", "cozy felted wool craft style"],
    "adjectives": ["cozy", "wizard", "steampunk", "detective", "astronaut", "chef", "gardener", "pirate", "scholar", "sleepy"],
    "accessories": ["wearing a tiny top hat", "wearing glowing brass goggles", "holding a miniature magical book", "wearing a soft knitted scarf", "wearing a tiny chef hat", "wearing an astronaut helmet", "holding a small wooden magnifying glass", "wearing a pirate eyepatch"],
    "actions": ["sitting on a branch", "reading a scroll", "drinking a cup of tea", "holding a golden star", "surrounded by magic runes", "staring with huge eyes"]
}

def generate_fallback_prompt():
    style = random.choice(FALLBACK_OWL_TEMPLATES["styles"])
    adj = random.choice(FALLBACK_OWL_TEMPLATES["adjectives"])
    acc = random.choice(FALLBACK_OWL_TEMPLATES["accessories"])
    act = random.choice(FALLBACK_OWL_TEMPLATES["actions"])
    return f"{style} of a {adj} owl, {acc}, {act}, clean solid background, high contrast, low resolution"

def get_ollama_prompt():
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma2:2b",
                "prompt": (
                    "Create a brief 1-sentence prompt for an image generator to draw a unique, cute owl character. "
                    "Use styles like 'low-res pixel art', 'claymation', or 'retro 8-bit game sprite'. "
                    "Include one funny accessory (e.g. wizard hat, goggles) and action. "
                    "End with 'clean solid background, low resolution'. Keep the description under 20 words."
                ),
                "stream": False
            },
            timeout=3.0
        )
        return response.json()["response"].strip().replace('"', '')
    except Exception:
        print("Ollama connection failed, using local prompt template generator.")
        return generate_fallback_prompt()

@app.get("/health")
def health():
    """Readiness probe — returns 200 once the model pipeline is loaded."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return {"status": "ready", "device": device}


@app.get("/generate-owl")
def generate_owl():
    if not pipeline:
        raise HTTPException(status_code=500, detail="Stable Diffusion pipeline is not loaded.")
    try:
        # Step 1: Get LLM-generated description
        prompt = get_ollama_prompt()
        print(f"Generated Prompt: {prompt}")
        
        # Step 2: Run Stable Diffusion (SD-Turbo works in exactly 1 step!)
        image = pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        
        # Downscale to represent a cute 256x256 character card
        image = image.resize((256, 256))
        
        # Step 3: Base64 Encode
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "prompt": prompt,
            "image": f"data:image/png;base64,{img_str}"
        }
    except Exception as e:
        print(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
