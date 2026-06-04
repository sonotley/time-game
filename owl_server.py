import base64
from io import BytesIO
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import StableDiffusionPipeline, LCMScheduler, AutoencoderTiny
import ollama
import string
import time

random.seed(time.time())

app = FastAPI(title="Owl Character Rewards API")

image_size = 512

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
    torch_dtype = torch.float32 # Stable for M1, prevents Metal driver crashes
elif torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

print(f"Loading Stable Diffusion onto device: {device.upper()}...", flush=True)

# 2. Load Dreamshaper-8-LCM (High quality + 4-step generation)
pipeline_load_error = None
try:
    model_id = "Lykon/dreamshaper-8-lcm" 

    # Load TAESD (Microscopic VAE for instant decoding)
    # Using AutoencoderTiny specifically for the TAESD architecture
    print("Loading TAESD VAE...", flush=True)
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch_dtype)

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, 
        vae=taesd,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        low_cpu_mem_usage=False, # Ensures stable loading without meta tensors
        device_map=None
    )

    # Use the baked-in LCM Scheduler
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

    # Aggressive memory management for 8GB Mac
    if device in ["cuda", "mps"]:
        # Re-enabling offloading to prevent system-level swapping on 8GB machines
        pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing()
    else:
        pipeline.to("cpu")
        
    print(f"Dreamshaper-8-LCM loaded successfully on {device.upper()}!", flush=True)
    pipeline.safety_checker = None

    # Step 2.5: Model Warmup (Pre-loads weights into GPU memory)
    if device == "mps":
        print("Warming up model...", flush=True)
        with torch.inference_mode():
            # Stable Speed: 4 steps, 2.0 guidance
            pipeline(prompt="warmup", num_inference_steps=1, guidance_scale=2.0, width=image_size, height=image_size)
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
    
    base_data = {
        "style": "Detailed 3d claymation",
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

def clean_llm_response(text: str, max_words: int = 20, mode: str = "story"):
    """
    Truncate at first line break and word limit.
    In 'story' mode, it attempts to end on a full sentence.
    """
    if not text: return ""

    # Remove common conversational intros
    fillers = ["Sure!", "Here is", "I'd be happy", "Certainly", "Ok, here", "Prompt:"]
    for filler in fillers:
        if text.lower().startswith(filler.lower()):
            parts = text.split(":", 1)
            if len(parts) > 1:
                text = parts[1]
            break

    # Take only the first non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines: return ""
    text = lines[0]

    words = text.split()
    if len(words) <= max_words:
        return text.replace('"', '').strip()

    # If over limit, truncate to max_words first
    truncated_text = " ".join(words[:max_words])

    if mode == "story":
        # Find the last sentence-ending punctuation in the truncated block
        last_punctuation = -1
        for char in [".", "!", "?"]:
            pos = truncated_text.rfind(char)
            if pos > last_punctuation:
                last_punctuation = pos
        
        # If we found punctuation that isn't at the very start (e.g., "Mr. Owl")
        if last_punctuation > 15:
            return truncated_text[:last_punctuation + 1].replace('"', '').strip()

    # Fallback for prompt mode or if no sentence end was found: standard ellipsis
    return truncated_text.replace('"', '').strip() + "..."

def embellish_owl_with_llm(traits: dict, story_max_words: int = 50, prompt_max_words: int = 35):
    try:
        # Call 1: The Story
        story_req = (
            f"Create a name and a fun back story for this owl, it must be no more then {story_max_words} words: "
            f"{traits['adjective']} owl, {traits['accessory']}, {traits['action']} at {traits['time_of_day']}.\n"
            f"No intro text. You *must* write the name first, then a colon ':', then the story. "
            f"The name *must* start with the letter {random.choice(string.ascii_letters)}"
        )
        print(f"LLM Pass 1 (Story) requesting...", flush=True)
        print(story_req)
        story_res = ollama.generate(
            model="llama3.2:1b", 
            prompt=story_req, 
            stream=False, 
            # keep_alive=0,
            options={
                "temperature": 0.9,  # Increases creativity/randomness
                "seed": random.randint(0, 999999) # Forces a new probability tree
            }
        )
        story = clean_llm_response(story_res['response'], max_words=story_max_words, mode="story")

        # Call 2: The Visuals
        visual_req = (
            f"Describe the appearance of this owl in a prompt suitable for stable diffusion "
            f"{traits['adjective']} owl, {traits['accessory']}, {traits['action']} at {traits['time_of_day']}.\n"
            f"Focus on visual details only. No intro text."
            f"Explicitly describe the basic physical details of the scene, what is in it, what colour, what size, what shape, how are the things positioned, etc."
            f"Here is some more detail about the owl: {story}."
            f"Do not use more than {prompt_max_words} words"
        )
        print(f"LLM Pass 2 (Visuals) requesting...", flush=True)
        print(visual_req)
        visual_res = ollama.generate(model="llama3.2:1b", prompt=visual_req, stream=False, keep_alive=0)
        embellished_prompt = clean_llm_response(visual_res['response'], max_words=prompt_max_words, mode="prompt")
        
        # Validation: check if we got meaningful text back
        if len(embellished_prompt) < 10 or len(story) < 5:
            print(f"LLM Validation failed. Story: '{story}', Prompt: '{embellished_prompt}'", flush=True)
            return None, None
            
        return embellished_prompt, story
        
    except Exception as e:
        print(f"Ollama embellishment failed: {e}", flush=True)
        return None, None

@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return {"status": "ready", "device": device}

@app.get("/generate-owl-info")
def generate_owl_info(time_of_day: str = "afternoon"):
    """Only generates the procedural traits and LLM embellishment (No SD)."""
    try:
        traits, fallback_prompt, fallback_story = generate_procedural_owl(time_of_day)
        embellished_prompt, story = embellish_owl_with_llm(traits)
        
        if embellished_prompt and story:
            style_name = traits.get('style', 'Detailed 3D Claymation')
            final_prompt = (
                f"high detail, masterpiece, clean background, vibrant colors. Owl, an owl, {style_name}. {embellished_prompt}, "
                f"set during the {time_of_day}"
            )
            final_story = story
        else:
            final_prompt = fallback_prompt
            final_story = fallback_story
            
        return {
            "prompt": final_prompt,
            "story": final_story
        }
    except Exception as e:
        print(f"Info generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-owl")
def generate_owl(time_of_day: str = "afternoon", prompt: str = None, story: str = None):
    print(f"Received owl generation request for {time_of_day}...", flush=True)
    if not pipeline:
        error_msg = f"Stable Diffusion pipeline is not loaded. Load error: {pipeline_load_error}"
        print(f"Error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)
    try:
        # Use provided prompt/story if available (pre-generated), otherwise generate now
        if prompt and story:
            print("Using pre-generated owl info.", flush=True)
            final_prompt = prompt
            final_story = story
        else:
            # Step 1: Generate procedural base
            traits, fallback_prompt, fallback_story = generate_procedural_owl(time_of_day)

            # Step 2: Embellish with LLM
            print("Embellishing owl with LLM...", flush=True)
            embellished_prompt, story = embellish_owl_with_llm(traits)

            # Use fallbacks if LLM fails, otherwise MANUALLY ENHANCE the visual prompt
            if embellished_prompt and story:
                # We add the style and quality keywords ourselves to ensure consistency
                style_name = traits.get('style', 'Detailed 3D Claymation')
                final_prompt = (
                    f"high detail, masterpiece, clean background, vibrant colors. Owl, an owl, {style_name}. {embellished_prompt}, "
                    f"set during the {time_of_day}"
                )
                final_story = story
            else:
                final_prompt = fallback_prompt
                final_story = fallback_story

        print(f"Final prompt for Diffusion: {final_prompt}", flush=True)
        print(f"Final story for UI: {final_story}", flush=True)

        # Step 3: Run Dreamshaper-8-LCM
        neg = "indistinct, flat, bad anatomy, deformed, blurry, low quality, distorted, extra limbs, bad hands, missing fingers, muddy textures, grainy, text, watermark"
        print("Starting Diffusion inference (this may take a while)...", flush=True)
        with torch.inference_mode():
            image = pipeline(
                prompt=final_prompt,
                negative_prompt=neg,
                num_inference_steps=4,     # Stable Speed: 4 steps in float32 is fast
                guidance_scale=2.0,        # Guidance scale for LCM should be low (1.0-2.0)
                width=image_size,             
                height=image_size
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