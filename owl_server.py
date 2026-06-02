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
    torch_dtype = torch.float16 # UNet speedup
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
    # Hybrid Precision: M1 chips need VAE in float32 to avoid black images (NaNs)
    # but the UNet runs significantly faster in float16.
    if device == "mps":
        pipeline.vae.to(dtype=torch.float32)

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

    # Step 2.5: Model Warmup (Pre-loads weights into GPU memory)
    if device == "mps":
        print("Warming up model...", flush=True)
        with torch.inference_mode():
            pipeline(prompt="warmup", num_inference_steps=1, width=128, height=128)
        print("Warmup complete!", flush=True)

except Exception as e:
    print(f"Error loading pipeline: {e}", flush=True)
    pipeline = None
    pipeline_load_error = str(e)

FALLBACK_OWL_TEMPLATES = {
    "adjectives": [
        "cozy", "wizard", "steampunk", "detective", "astronaut", "chef", "gardener", "pirate", "scholar", "sleepy",
        "grumpy", "majestic", "spunky", "bashful", "elegant", "clumsy", "sporty", "brave", "curious", "dapper",
        "eccentric", "friendly", "gloomy", "heroic", "mischievous", "puzzled", "radiant", "studious", "timid", "wild"
    ],
    "accessories": [
        "wearing a tiny top hat", "wearing glowing brass goggles", "holding a miniature magical book", 
        "wearing a soft knitted scarf", "wearing a tiny chef hat", "wearing an astronaut helmet", 
        "holding a small wooden magnifying glass", "wearing a pirate eyepatch", "wearing a flower crown",
        "carrying a tiny backpack", "holding a glowing lantern", "wearing a bright red bowtie",
        "wearing a oversized sweater", "holding a golden key", "wearing a silver monocle",
        "carrying a small wicker basket", "wearing a yellow raincoat", "holding a single colorful feather",
        "wearing a string of pearls", "holding a miniature telescope", "wearing a fuzzy earmuff",
        "carrying a small compass", "wearing a tiny superhero cape", "holding a small paintbrush"
    ],
    "actions": [
        "sitting on a branch", "reading a scroll", "drinking a cup of tea", "holding a golden star", 
        "surrounded by magic runes", "staring with huge eyes", "balancing on a stack of books",
        "peering out of a hollow tree", "tending to a small sprout", "examining a clock gear",
        "waving a tiny flag", "polishing a gemstone", "sketching in a notebook", "listening to a seashell",
        "adjusting their glasses", "counting some silver coins", "nibbling on a cracker", "sleeping on a cloud",
        "playing a tiny flute", "mixing a potion", "braiding some straw", "gazing at a compass"
    ]
}

def generate_procedural_owl(time_of_day: str):
    adj = random.choice(FALLBACK_OWL_TEMPLATES["adjectives"])
    acc = random.choice(FALLBACK_OWL_TEMPLATES["accessories"])
    act = random.choice(FALLBACK_OWL_TEMPLATES["actions"])
    
    # Fixed style: Detailed 3D Claymation
    base_data = {
        "style": "Detailed 3D Claymation",
        "adjective": adj,
        "accessory": acc,
        "action": act,
        "time_of_day": time_of_day
    }
    
    # Simple fallback prompt for SD if LLM fails
    fallback_prompt = (
        f"Detailed 3D claymation of a {adj} owl, {acc}, {act}, "
        f"set during the {time_of_day}, clean solid background, high detail, masterpiece"
    )
    # Simple fallback story for UI if LLM fails
    fallback_story = f"A {adj} owl is {act}."
    
    return base_data, fallback_prompt, fallback_story

def embellish_owl_with_llm(traits: dict):
    try:
        # Prompt Gemma to embellish the visual details and create a character
        llm_prompt = (
            f"Task: Create a character story and visual description for a 3D claymation owl.\n"
            f"Base Traits: {traits['adjective']} owl, {traits['accessory']}, {traits['action']}.\n"
            f"Environment: {traits['time_of_day']}.\n"
            f"Art Style: Detailed 3D Claymation (tactile textures, clay fingerprints, lighting matching the {traits['time_of_day']}).\n"
            f"Instructions:\n"
            f"- Output exactly two lines.\n"
            f"- Line 1 must start with 'PROMPT:' followed by a detailed visual prompt for an image generator.\n"
            f"- Line 2 must start with 'STORY:' followed by a name and a charming sentence about the owl.\n"
            f"Example:\n"
            f"PROMPT: A tactile 3D claymation owl with fingerprint textures...\n"
            f"STORY: Barnaby the Wise is reading a tiny scroll..."
        )
        
        response = ollama.generate(model="llama3.2", prompt=llm_prompt, stream=False)
        text = response['response'].strip()
        
        embellished_prompt = ""
        story = ""
        
        # Robust parsing for labels like **PROMPT:**, PROMPT:, Prompt:, etc.
        for line in text.split('\n'):
            clean_line = line.strip().replace("*", "")
            if clean_line.upper().startswith("PROMPT:"):
                embellished_prompt = clean_line[len("PROMPT:"):].strip()
            elif clean_line.upper().startswith("STORY:"):
                story = clean_line[len("STORY:"):].strip()
        
        # Validation
        if not embellished_prompt or not story:
            print(f"Parsing failed for LLM output:\n{text}", flush=True)
            raise ValueError("LLM response format invalid")
            
        return embellished_prompt, story
        
    except Exception as e:
        print(f"Ollama embellishment failed: {e}", flush=True)
        return None, None

@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return {"status": "ready", "device": device}


@app.get("/generate-owl")
def generate_owl(time_of_day: str = "afternoon"):
    print(f"Received owl generation request for {time_of_day}...", flush=True)
    if not pipeline:
        error_msg = f"Stable Diffusion pipeline is not loaded. Load error: {pipeline_load_error}"
        print(f"Error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)
    try:
        # Step 1: Generate procedural base
        traits, fallback_prompt, fallback_story = generate_procedural_owl(time_of_day)

        # Step 2: Embellish with LLM
        print("Embellishing owl with LLM...", flush=True)
        embellished_prompt, story = embellish_owl_with_llm(traits)

        # Use fallbacks if LLM fails
        final_prompt = embellished_prompt if embellished_prompt else fallback_prompt
        final_story = story if story else fallback_story

        print(f"Final prompt for Diffusion: {final_prompt}", flush=True)
        print(f"Final story for UI: {final_story}", flush=True)

        # Step 3: Run Tiny-SD 
        print("Starting Diffusion inference (this may take a while)...", flush=True)
        with torch.inference_mode():
            image = pipeline(
                prompt=final_prompt, 
                num_inference_steps=10,
                guidance_scale=7.5,    
                width=256,             
                height=256
            ).images[0]
        print("Inference complete!", flush=True)

        # Step 4: Base64 Encode output
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("Returning generated owl.", flush=True)
        return {
            "prompt": final_prompt,
            "story": final_story,
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