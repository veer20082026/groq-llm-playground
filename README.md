# ⚡ Groq LLM Playground: Zero-Shot, One-Shot & Few-Shot Prompt Analyzer

A production-ready LLM playground powered by **Groq LPU™** that analyzes prompts in real-time, categorizes them as **Zero-shot**, **One-shot**, or **Few-shot**, provides deep prompt engineering diagnostics, and executes them with ultra-low latency.

---

## 🌟 Key Features

1. **Dual-Engine Shot Classification**:
   - **Real-Time Heuristic Detector**: Automatically counts demonstrations, parses Input/Output pairs, Q&A blocks, and arrow mappings (`A -> B`) as you type with 0 latency.
   - **Deep Groq LLM Semantic Diagnostics**: Evaluates the prompt structure using models like `llama-3.3-70b-versatile`, extracts exact demonstration examples, computes a Prompt Quality Score (1-100), and flags prompt techniques (e.g., Chain-of-Thought, Delimiters, Role Persona).
2. **Groq Model Playground**:
   - Test prompts on flagship models:
     - `llama-3.3-70b-versatile` (128k context, state-of-the-art open reasoning)
     - `llama-3.1-8b-instant` (800+ tokens/sec)
     - `mixtral-8x7b-32768` (32k context MoE)
     - `gemma2-9b-it` (Google lightweight model)
   - Real-time **Groq Performance HUD**: measures inference speed (`tokens/sec`), latency (`ms`), and token counts.
   - Fine-tune Temperature, Max Output Tokens, Top-P, and System Prompts.
3. **In-Context Learning Prompt Converter**:
   - One-click automatic transformation of any Zero-shot prompt into a high-quality One-shot or 3-shot Few-shot prompt.
4. **Preloaded Prompt Library**:
   - Built-in curated presets across Classification, Entity Extraction, Translation, Code Generation, and Chain-of-Thought Math.
5. **Production & Cloud Ready**:
   - Includes health check endpoints (`/health`, `/api/health`).
   - WSGI support via `wsgi.py` and `gunicorn`.
   - Ready-to-deploy configuration files for **Render**, **Railway**, **Vercel**, and **Docker**.

---

## 🌐 Deploying as a Website (Live Deployment Guide)

You can deploy this application to any cloud platform in under 2 minutes:

### Option 1: Render (Recommended - Free Tier)
1. Push this repository to **GitHub** or **GitLab**.
2. Go to [render.com](https://render.com) and click **New +** → **Web Service**.
3. Connect your repository.
4. Render automatically detects the configuration:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
5. *(Optional)* In **Environment Variables**, add:
   - `GROQ_API_KEY`: `your_groq_api_key_here`
6. Click **Deploy Web Service**. Your playground is live on a free `onrender.com` URL!

---

### Option 2: Railway
1. Go to [railway.app](https://railway.app) and select **New Project** → **Deploy from GitHub repo**.
2. Select this repository. Railway automatically reads [railway.json](railway.json) and starts the app.
3. Add `GROQ_API_KEY` under **Variables** in the Railway dashboard.
4. Generate a public domain under **Settings** → **Networking**.

---

### Option 3: Vercel
1. Install Vercel CLI or connect via the Vercel dashboard:
   ```bash
   npm i -g vercel
   vercel
   ```
2. Vercel will automatically read [vercel.json](vercel.json) and deploy `wsgi.py` as a serverless Python function.

---

### Option 4: Docker / Cloud VPS (AWS, GCP, DigitalOcean, Hetzner)
Build and run the containerized application with a single command:
```bash
# Using Docker Compose
docker compose up -d --build
```
Or build manually:
```bash
docker build -t groq-playground .
docker run -d -p 5000:5000 -e GROQ_API_KEY="your_key" groq-playground
```

---

## 💻 Local Quick Start

### 1. Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Set Your Groq API Key (Optional)
You can set your API key as an environment variable or enter it directly in the Web UI:
```bash
export GROQ_API_KEY="gsk_your_groq_api_key_here"
```
*(Get a free key at [console.groq.com/keys](https://console.groq.com/keys))*

### 3. Launch the Web Playground
```bash
python3 app.py
```
Open your browser at **[http://localhost:5000](http://localhost:5000)**.

---

## 🖥️ Terminal CLI Usage

You can also analyze prompts directly from your terminal:

```bash
# Analyze a prompt string
python3 cli.py "Review: Great screen. Sentiment: Positive
Review: Battery dies fast. Sentiment: Negative
Review: Loved the audio quality! Sentiment:"

# Output raw JSON
python3 cli.py "Translate 'Hello' to Spanish." --json
```

---

## 📖 Prompt Shot Definitions

| Type | Solved Demonstrations | Purpose |
| :--- | :--- | :--- |
| **Zero-Shot** | **0** | Direct instruction or query with no prior solved examples. |
| **One-Shot** | **1** | Exactly 1 demonstration pair to demonstrate expected formatting or reasoning. |
| **Few-Shot** | **≥ 2** | Two or more demonstration pairs for in-context pattern learning. |

---

## 📁 Project Structure

```
llm playground/
├── app.py                 # HTTP Server & REST API endpoints
├── wsgi.py                # Production WSGI application adapter
├── cli.py                 # Terminal CLI for quick prompt checking
├── detector.py            # Real-time regex & heuristic shot classifier
├── groq_service.py        # Groq client integration (inference, diagnostics, transformations)
├── static/
│   ├── index.html         # Modern dark-mode web interface
│   ├── style.css          # Sleek styling, HUD, badges & tabs
│   └── app.js             # Live client-side logic & reactivity
├── requirements.txt       # Production dependencies
├── Dockerfile             # Container configuration with healthcheck
├── docker-compose.yml     # Multi-platform container setup
├── Procfile               # Cloud process starter (Render / Railway / Heroku)
├── render.yaml            # 1-click configuration for Render.com
├── railway.json           # 1-click configuration for Railway.app
├── vercel.json            # Serverless deployment configuration for Vercel
├── .env.example           # Example environment file
└── README.md              # Project documentation
```
