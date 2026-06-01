import base64
from io import BytesIO
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import ollama

app = FastAPI(title="Owl Character Rewards API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Device and Budget Memory Selection
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32 
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

print(f"Loading Stable Diffusion onto device: {device.upper()}...")

# 2. Load Tiny-SD (Optimized for 8GB RAM and speed)
pipeline_load_error = None
try:
    model_id = "segmind/Tiny-SD" 
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        use_safetensors=False
    )
    # Tiny-SD works much better with DPM++ scheduler for few-step generation
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # Aggressive memory optimization configurations for tight 8GB setups
    if device in ["cuda", "mps"]:
        pipeline.to(device)
        pipeline.enable_attention_slicing()
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cpu")
        
    print(f"Tiny-SD loaded successfully on {device.upper()}!", flush=True)
    pipeline.safety_checker = None
except Exception as e:
    print(f"Error loading pipeline: {e}", flush=True)
    pipeline = None
    pipeline_load_error = str(e)

FALLBACK_OWL_TEMPLATES = {
    "styles": ["vibrant oil painting", "cute 3D claymation", "detailed digital art", "cozy felted wool craft style", "chibi watercolor"],
    "adjectives": ["cozy", "wizard", "steampunk", "detective", "astronaut", "chef", "gardener", "pirate", "scholar", "sleepy"],
    "accessories": ["wearing a tiny top hat", "wearing glowing brass goggles", "holding a miniature magical book", "wearing a soft knitted scarf", "wearing a tiny chef hat", "wearing an astronaut helmet", "holding a small wooden magnifying glass", "wearing a pirate eyepatch"],
    "actions": ["sitting on a branch", "reading a scroll", "drinking a cup of tea", "holding a golden star", "surrounded by magic runes", "staring with huge eyes"]
}

def generate_fallback_prompt():
    style = random.choice(FALLBACK_OWL_TEMPLATES["styles"])
    adj = random.choice(FALLBACK_OWL_TEMPLATES["adjectives"])
    acc = random.choice(FALLBACK_OWL_TEMPLATES["accessories"])
    act = random.choice(FALLBACK_OWL_TEMPLATES["actions"])
    return f"{style} of a {adj} owl, {acc}, {act}, clean solid background, high detail, masterpiece"

def get_ollama_prompt():
    try:
        # Use a more descriptive and strict prompt for Gemma
        response = ollama.generate(
            model="gemma:2b",
            prompt=(
                "Task: Write a high-quality image generation prompt for a cute owl character.\n"
                "Constraints:\n"
                "- One sentence only.\n"
                "- Include a specific artistic style (e.g., '3D claymation', 'vibrant digital art').\n"
                "- Include one funny accessory and one action.\n"
                "- End with 'clean solid background, high detail'.\n"
                "- Do NOT use any introductory text or conversational filler.\n"
                "Example: A cute 3D claymation owl wearing a tiny chef hat and holding a spatula, clean solid background, high detail."
            ),
            stream=False
        )
        # Clean up the response to ensure no extra text
        lines = response['response'].strip().split('\n')
        prompt = lines[-1].strip().replace('"', '')
        if len(prompt) < 10: prompt = generate_fallback_prompt()
        return prompt
    except Exception as e:
        # Diagnostic printing to catch precisely why the framework failed
        print(f"Ollama connection or execution failed: {e}")
        print("Falling back to local template generator.")
        return generate_fallback_prompt()

@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return {"status": "ready", "device": device}


@app.get("/generate-owl")
def generate_owl():
    print("Received owl generation request...", flush=True)
    if not pipeline:
        error_msg = f"Stable Diffusion pipeline is not loaded. Load error: {pipeline_load_error}"
        print(f"Error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)
    try:
        # Step 1: Securely obtain the text prompt string
        print("Generating prompt with Ollama...", flush=True)
        prompt = get_ollama_prompt()
        print(f"Final prompt for Diffusion: {prompt}", flush=True)
        
        # Step 2: Run Tiny-SD 
        print("Starting Diffusion inference (this may take a while)...", flush=True)
        with torch.inference_mode():
            image = pipeline(
                prompt=prompt, 
                num_inference_steps=12, # Increased for better clarity
                guidance_scale=7.5,    
                width=256,             
                height=256
            ).images[0]
        print("Inference complete!", flush=True)
        
        # Step 3: Base64 Encode output
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print("Returning generated owl.", flush=True)
        return {
            "prompt": prompt,
            "image": f"data:image/png;base64,{img_str}"
        }
    except Exception as e:
        error_msg = f"Generation failed during inference: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    print("Starting Owl Server on http://0.0.0.0:8081")
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")