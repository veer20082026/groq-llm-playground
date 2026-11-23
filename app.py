"""
Groq LLM Playground Backend Server
A lightweight, zero-external-dependency web server built on Python standard library.
Serves the web playground UI and REST APIs for prompt classification and Groq execution.
"""

import os
import json
import mimetypes
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from detector import PromptShotDetector
from groq_service import GroqService, SUPPORTED_MODELS, GROQ_AVAILABLE

PORT = int(os.getenv("PORT", "5000"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Sample presets for quick testing in the playground
PRESET_PROMPTS = [
    {
        "id": "sentiment-zero",
        "title": "Zero-Shot: Sentiment Analysis",
        "category": "Classification",
        "expected_shot": "zero-shot",
        "description": "Standard zero-shot classification prompt with no solved examples.",
        "system_prompt": "You are a concise text classification assistant.",
        "prompt": "Classify the sentiment of the following customer review as either POSITIVE, NEGATIVE, or NEUTRAL.\n\nReview: The noise cancelling on these headphones is miraculous, though the ear cushions get slightly warm after 3 hours of continuous use.\n\nSentiment:"
    },
    {
        "id": "sentiment-one",
        "title": "One-Shot: Sentiment Analysis",
        "category": "Classification",
        "expected_shot": "one-shot",
        "description": "One-shot prompt providing exactly 1 demonstration input/output pair.",
        "system_prompt": "You are a concise text classification assistant.",
        "prompt": "Classify the sentiment of customer reviews.\n\nExample 1:\nReview: The battery died within 2 hours and customer service never responded.\nSentiment: NEGATIVE (Confidence: 0.99, Key issue: battery life & support)\n\nNow classify this review:\nReview: The display is stunningly bright in direct sunlight, and shipping was next-day!\nSentiment:"
    },
    {
        "id": "entity-few",
        "title": "Few-Shot: Structured Entity Extraction",
        "category": "Extraction",
        "expected_shot": "few-shot",
        "description": "Few-shot prompt with 3 demonstrations teaching JSON output formatting.",
        "system_prompt": "Extract entities into strict JSON format.",
        "prompt": "Extract person names, companies, and locations from text into JSON.\n\nInput: Tim Cook announced new hardware at the Steve Jobs Theater in Cupertino.\nOutput: {\"person\": [\"Tim Cook\"], \"company\": [\"Apple\"], \"location\": [\"Cupertino\", \"Steve Jobs Theater\"]}\n\nInput: Satya Nadella visited Berlin to discuss cloud infrastructure for Siemens.\nOutput: {\"person\": [\"Satya Nadella\"], \"company\": [\"Microsoft\", \"Siemens\"], \"location\": [\"Berlin\"]}\n\nInput: Jensen Huang showed the new Blackwell GPUs during Computex in Taipei.\nOutput: {\"person\": [\"Jensen Huang\"], \"company\": [\"NVIDIA\"], \"location\": [\"Taipei\"]}\n\nInput: Demis Hassabis delivered a keynote on AlphaFold at Oxford University.\nOutput:"
    },
    {
        "id": "cot-few",
        "title": "Few-Shot: Chain-of-Thought Math",
        "category": "Reasoning",
        "expected_shot": "few-shot",
        "description": "Few-shot prompt teaching step-by-step reasoning (Chain-of-Thought).",
        "system_prompt": "Solve word problems with step-by-step reasoning.",
        "prompt": "Solve the math problem with step-by-step reasoning before stating the final answer.\n\nQ: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\nA: Roger started with 5 balls. 2 cans of 3 balls each is 2 * 3 = 6 tennis balls. 5 + 6 = 11. Therefore, Roger has 11 tennis balls.\n\nQ: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?\nA: The cafeteria had 23 apples originally. They used 20, so they had 23 - 20 = 3 apples. Then they bought 6 more, so they now have 3 + 6 = 9 apples. Therefore, the answer is 9.\n\nQ: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?\nA:"
    },
    {
        "id": "code-zero",
        "title": "Zero-Shot: Code Generation",
        "category": "Coding",
        "expected_shot": "zero-shot",
        "description": "Zero-shot instruction requesting a Python helper function.",
        "system_prompt": "You are an expert Python engineer.",
        "prompt": "Write a clean, production-ready Python generator function called `chunk_iterable(iterable, chunk_size)` that yields slices of the given iterable. Include type hints and docstrings."
    },
    {
        "id": "arrow-one",
        "title": "One-Shot: Arrow Translation",
        "category": "Translation",
        "expected_shot": "one-shot",
        "description": "One-shot prompt using concise arrow mapping notation.",
        "system_prompt": "Translate English idioms to equivalent French idioms.",
        "prompt": "Translate English idioms into French equivalents:\nOnce in a blue moon -> Tous les 36 du mois\n\nWhen pigs fly ->"
    }
]


class PlaygroundRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for convenience
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Groq-API-Key")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_api_key_from_headers(self) -> str:
        key = self.headers.get("X-Groq-API-Key")
        if not key:
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                key = auth[7:].strip()
        return key or os.getenv("GROQ_API_KEY", "")

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # REST API Routes
        if path in ("/health", "/api/health"):
            self._send_json(200, {"status": "healthy", "service": "groq-llm-playground"})
            return

        if path == "/api/status":
            env_key = os.getenv("GROQ_API_KEY", "")
            masked_key = f"{env_key[:4]}...{env_key[-4:]}" if len(env_key) > 8 else ("Set" if env_key else "Not set")
            self._send_json(200, {
                "ok": True,
                "groq_package_installed": GROQ_AVAILABLE,
                "has_env_key": bool(env_key),
                "masked_env_key": masked_key,
                "models": SUPPORTED_MODELS
            })
            return

        if path == "/api/presets":
            self._send_json(200, {
                "presets": PRESET_PROMPTS
            })
            return

        # Serve static files
        if path == "/" or path == "":
            file_path = os.path.join(STATIC_DIR, "index.html")
        else:
            clean_path = path.lstrip("/")
            file_path = os.path.join(STATIC_DIR, clean_path)

        # Security check against directory traversal
        try:
            real_file = os.path.realpath(file_path)
            real_static = os.path.realpath(STATIC_DIR)
            if not real_file.startswith(real_static):
                self.send_error(403, "Access Denied")
                return
        except Exception:
            self.send_error(404, "File Not Found")
            return

        if os.path.exists(file_path) and os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, f"File Not Found: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON payload: {str(e)}"})
            return

        api_key = body.get("api_key") or self._get_api_key_from_headers()
        groq_service = GroqService(api_key=api_key)

        # 1. Key validation test
        if path == "/api/test-key":
            res = groq_service.test_connection(api_key=api_key)
            self._send_json(200 if res.get("success") else 400, res)
            return

        # 2. Heuristic prompt analyzer (fast, local, no API key required)
        if path == "/api/analyze-heuristic":
            prompt = body.get("prompt", "")
            result = PromptShotDetector.analyze(prompt)
            self._send_json(200, result)
            return

        # 3. Deep Groq LLM prompt analyzer
        if path == "/api/analyze-groq":
            prompt = body.get("prompt", "")
            model = body.get("model", "llama-3.3-70b-versatile")
            if not prompt.strip():
                self._send_json(400, {"error": "Prompt cannot be empty."})
                return
            try:
                # First run heuristic
                heuristic = PromptShotDetector.analyze(prompt)
                # Then run deep Groq analysis
                deep_analysis = groq_service.analyze_prompt_deep(prompt, api_key=api_key, model=model)
                self._send_json(200, {
                    "heuristic": heuristic,
                    "deep": deep_analysis
                })
            except Exception as e:
                # If Groq call fails (e.g. invalid key or network), still return heuristic with warning
                heuristic = PromptShotDetector.analyze(prompt)
                self._send_json(200, {
                    "heuristic": heuristic,
                    "deep": None,
                    "groq_error": str(e)
                })
            return

        # 4. Execute prompt on Groq
        if path == "/api/execute":
            prompt = body.get("prompt", "")
            system_prompt = body.get("system_prompt", "")
            model = body.get("model", "llama-3.3-70b-versatile")
            temperature = float(body.get("temperature", 0.7))
            max_tokens = int(body.get("max_tokens", 1024))
            top_p = float(body.get("top_p", 1.0))

            if not prompt.strip():
                self._send_json(400, {"error": "Prompt cannot be empty."})
                return

            try:
                res = groq_service.execute_prompt(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    api_key=api_key
                )
                self._send_json(200, res)
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})
            return

        # 5. Transform prompt (Zero -> One/Few-shot)
        if path == "/api/transform":
            prompt = body.get("prompt", "")
            target_shot = body.get("target_shot", "few-shot")
            model = body.get("model", "llama-3.3-70b-versatile")

            if not prompt.strip():
                self._send_json(400, {"error": "Prompt cannot be empty."})
                return

            try:
                res = groq_service.transform_prompt(
                    prompt=prompt,
                    target_shot=target_shot,
                    api_key=api_key,
                    model=model
                )
                self._send_json(200, res)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        self._send_json(404, {"error": f"API route not found: {path}"})

    def log_message(self, format, *args):
        # Concise logging
        print(f"[{self.log_date_time_string()}] {self.command} {self.path} - {args[0] if args else ''}")


def start_server(host: str = "0.0.0.0", port: int = PORT):
    os.makedirs(STATIC_DIR, exist_ok=True)
    server_address = (host, port)
    httpd = HTTPServer(server_address, PlaygroundRequestHandler)
    print(f"🚀 Groq LLM Playground running at: http://{host}:{port}")
    print(f"📁 Serving static files from: {STATIC_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
