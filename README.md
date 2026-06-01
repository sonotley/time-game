# Tell the Time

An interactive HTML5 game that teaches children to read analogue clocks using a guided 4-step method. Perfect rounds hatch unique AI-generated owl characters as rewards.

## Playing the game (no setup required)

The game is a single self-contained file — just open `index.html` in a browser, or serve it locally:

```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

When the AI owl server is not running, the game automatically falls back to procedurally generated vector owl characters, so it is fully playable without any Python setup.

---

## AI Owl Rewards (optional)

Completing a round without any mistakes hatches a unique AI-generated owl character. This requires two local services:

### 1. Ollama (prompt generation)

Download from [ollama.com](https://ollama.com), open the app so the menu-bar icon appears, then pull the model:

```bash
ollama pull gemma2:2b
```

### 2. Python generation server (image generation)

Requires Python 3.9+ with PyTorch. On Apple Silicon the model runs on the Metal GPU (MPS).

```bash
pip install -r requirements.txt
python3 owl_server.py
```

The server starts on **http://127.0.0.1:8081**. It takes ~20–30 seconds to load Stable Diffusion Turbo the first time. You can check it is ready at:

```
http://127.0.0.1:8081/health
```

Once it returns `{"status":"ready"}`, start a game at **http://localhost:8080** and complete a round perfectly to hatch your first owl.

---

## How the game works

The clock-reading skill is broken into four scaffolded steps:

| Step | What the child identifies |
|------|--------------------------|
| 1 | Which two numbers the **short (hour) hand** is between |
| 2 | Whether the **long (minute) hand** is on the *past* or *to* side |
| 3 | How many minutes (by name) along the arc |
| 4 | The assembled phrase, e.g. *"quarter past three"* |

A developing phrase builder (`? • past/to • ?`) fills in as each step is answered, showing how the final time phrase is constructed.

---

## Project structure

| File | Description |
|------|-------------|
| `index.html` | Complete self-contained game (HTML + CSS + JS) |
| `owl_server.py` | FastAPI server — generates owl images via SD-Turbo + Ollama |
| `requirements.txt` | Python dependencies for the owl server |
