import base64
from io import BytesIO
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler 
from python_coreml_stable_diffusion.pipeline import get_coreml_pipe
import ollama
import string
import time

app = FastAPI(title="Owl Character Rewards API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ART_STYLE = "Fantasy art"

# 1. Core ML Pipeline Setup
print("Loading Core ML Pipeline to Apple Neural Engine (ANE)...", flush=True)

pipeline_load_error = None
try:
    # 1. Load the base PyTorch configuration (required for tokenizer/scheduler logic)
    print("Loading base PyTorch configuration...", flush=True)
    pytorch_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        use_safetensors=True
    )
    pytorch_pipe.safety_checker = None

    print("Swapping scheduler to DPM++ 2M Karras...", flush=True)
    pytorch_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pytorch_pipe.scheduler.config, 
        use_karras_sigmas=True
    )

    # 2. Wrap it with Apple's Core ML backend using the downloaded local models
    # We point to the specific 'compiled' directory to avoid ambiguity errors
    print("Wrapping with Core ML backend (models2/DreamShaper-v8_split-einsum_cn)...", flush=True)
    pipeline = get_coreml_pipe(
        pytorch_pipe=pytorch_pipe,
        mlpackages_dir="./models2/DreamShaper-v8_split-einsum_cn",
        model_version="runwayml/stable-diffusion-v1-5",
        compute_unit="ALL" # Targets CPU + GPU + Neural Engine simultaneously
    )
    
    print("Core ML Pipeline successfully loaded!", flush=True)

except Exception as e:
    print(f"Error loading Core ML pipeline: {e}", flush=True)
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
        "style": ART_STYLE,
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
    Cleans LLM response, joining lines and intelligently truncating at word limits.
    In 'story' mode, it prefers ending at a full sentence boundary.
    """
    if not text: return ""

    # 1. Join all non-empty lines with a space to prevent loss of multi-line stories
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = " ".join(lines)

    # 2. Remove common conversational intros (e.g., "Sure! Here's your owl:")
    fillers = ["Sure!", "Here is", "I'd be happy", "Certainly", "Ok, here", "Prompt:"]
    for filler in fillers:
        if text.lower().startswith(filler.lower()):
            # Try to find a colon or a clean break after the filler
            first_colon = text.find(":")
            if 0 < first_colon < 100: # Found a colon that likely separates intro from content
                text = text[first_colon+1:].strip()
            break

    # 3. Check if it even exceeds the word count
    words = text.split()
    if len(words) <= max_words:
        return text.replace('"', '').strip()

    # 4. Handle Truncation
    if mode == "story":
        # Find all punctuation marks that look like sentence ends
        endings = []
        for i, char in enumerate(text):
            if char in [".", "!", "?"]:
                # Check if it's a real end (followed by space or end of string)
                if i + 1 == len(text) or text[i+1].isspace():
                    word_count = len(text[:i+1].split())
                    endings.append((i, word_count))
        
        if endings:
            # Find the ending closest to our word limit
            # (We prefer slightly under, but will take slightly over if it's closer)
            best_idx, _ = min(endings, key=lambda x: abs(x[1] - max_words))
            
            # Sanity check: don't pick an ending that is too far beyond the limit
            if len(text[:best_idx+1].split()) <= max_words + 10:
                return text[:best_idx+1].replace('"', '').strip()

    # Fallback for prompt mode or if no suitable sentence end was found
    truncated = " ".join(words[:max_words])
    return truncated.replace('"', '').strip() + "..."

def get_ollama_model() -> str:
    """Finds an installed Ollama model matching preferred candidates, with fallback."""
    preferred_models = ["llama3.2:1b"]
    try:
        response = ollama.list()
        available = [m.model for m in getattr(response, "models", [])]
        for pm in preferred_models:
            if pm in available:
                return pm
            for am in available:
                if am.startswith(pm) or pm.startswith(am):
                    return am
        if available:
            return available[0]
    except Exception as e:
        print(f"Warning: Failed to list Ollama models ({e}). Using default fallback.")
    return "llama3.2:1b"

def embellish_owl_with_llm(traits: dict, story_max_words: int = 50, prompt_max_words: int = 35):
    try:
        model = get_ollama_model()
        prompt = (
            f"Create a character profile for an owl with these traits:\n"
            f"- Adjective: {traits['adjective']}\n"
            f"- Accessory: {traits['accessory']}\n"
            f"- Action: {traits['action']}\n"
            f"- Time of day: {traits['time_of_day']}\n\n"
            f"Respond ONLY with a JSON object containing these exact fields:\n"
            f"- 'name': A creative name for the owl starting with the letter '{random.choice(string.ascii_uppercase)}'\n"
            f"- 'story': A fun backstory of at most {story_max_words} words. At least {story_max_words-10} words. \n"
            f"- 'visual_prompt': A detailed visual description of approximately {prompt_max_words} words, suitable for Stable Diffusion, focusing on physical details, shapes, colors, positioning, and style. Do not mention the name in the prompt.\n"
        )
        
        print(f"LLM single-pass request to model '{model}'...", flush=True)
        res = ollama.generate(
            model=model, 
            prompt=prompt,
            format="json",
            stream=False, 
            keep_alive=0,
            options={
                "temperature": 0.85,
                "seed": random.randint(0, 999999)
            }
        )
        
        data = json.loads(res['response'].strip())
        name = data.get('name', 'Mysterious Owl').strip()
        story_content = data.get('story', '').strip()
        visual_prompt = data.get('visual_prompt', '').strip()
        
        combined_story = f"{name}: {story_content}" if story_content else name
        
        if len(visual_prompt) < 10 or len(combined_story) < 5:
            print(f"LLM Validation failed. Story: '{combined_story}', Prompt: '{visual_prompt}'", flush=True)
            return None, None
            
        return visual_prompt, combined_story
        
    except Exception as e:
        print(f"Ollama embellishment failed: {e}", flush=True)
        return None, None

@app.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return {"status": "ready", "device": "Core ML"}

@app.get("/generate-owl-info")
def generate_owl_info(time_of_day: str = "afternoon"):
    """Only generates the procedural traits and LLM embellishment (No SD)."""
    try:
        traits, fallback_prompt, fallback_story = generate_procedural_owl(time_of_day)
        embellished_prompt, story = embellish_owl_with_llm(traits)
        
        if embellished_prompt and story:
            style_name = traits.get('style', 'Detailed 3D Claymation')
            final_prompt = (
                # f"high detail, masterpiece, clean background, vibrant colors. "
                f"An owl character on a trading card, "
                f"{style_name}. {embellished_prompt}, "
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

        # Step 3: Run Core ML Stable Diffusion
        print("Starting Core ML inference...", flush=True)
        neg = "borders, text, scary, obscene, boring, nsfw, not an owl, human, oversaturated, deformed, bad anatomy, bad proportions, blurry, low quality, worst quality, artifacts, noise, text, watermark, mutated, extra limbs, fused fingers"
        # Inference resolution is locked to model bundle (512x512)
        image = pipeline(
            prompt=final_prompt,
            negative_prompt = neg,
            height=pipeline.height, 
            width=pipeline.width,
            num_inference_steps=20, # Standard steps for Core ML v1.5
            guidance_scale=7.5
        ).images[0]
        print("Core ML Inference complete!", flush=True)

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