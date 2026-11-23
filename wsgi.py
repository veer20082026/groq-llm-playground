"""
WSGI Application Adapter for Production Deployment (Gunicorn, uWSGI, Render, Railway, etc.)
Allows running Groq LLM Playground with production WSGI servers:
    gunicorn wsgi:application --workers 2 --bind 0.0.0.0:$PORT
"""

import os
import json
import mimetypes
from urllib.parse import parse_qs

from detector import PromptShotDetector
from groq_service import GroqService, SUPPORTED_MODELS, GROQ_AVAILABLE
from app import PRESET_PROMPTS, STATIC_DIR

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")

    # CORS Headers
    headers = [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Groq-API-Key"),
    ]

    if method == "OPTIONS":
        start_response("204 No Content", headers)
        return [b""]

    # 1. Health check endpoints
    if method == "GET" and path in ("/health", "/api/health"):
        body = json.dumps({"status": "healthy", "service": "groq-llm-playground"}).encode("utf-8")
        headers.extend([
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body)))
        ])
        start_response("200 OK", headers)
        return [body]

    # 2. Status API
    if method == "GET" and path == "/api/status":
        env_key = os.getenv("GROQ_API_KEY", "")
        masked_key = f"{env_key[:4]}...{env_key[-4:]}" if len(env_key) > 8 else ("Set" if env_key else "Not set")
        payload = {
            "ok": True,
            "groq_package_installed": GROQ_AVAILABLE,
            "has_env_key": bool(env_key),
            "masked_env_key": masked_key,
            "models": SUPPORTED_MODELS
        }
        body = json.dumps(payload, indent=2).encode("utf-8")
        headers.extend([
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body)))
        ])
        start_response("200 OK", headers)
        return [body]

    # 3. Presets API
    if method == "GET" and path == "/api/presets":
        body = json.dumps({"presets": PRESET_PROMPTS}, indent=2).encode("utf-8")
        headers.extend([
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body)))
        ])
        start_response("200 OK", headers)
        return [body]

    # 4. POST Endpoints
    if method == "POST":
        content_length = int(environ.get("CONTENT_LENGTH", 0) or 0)
        request_body = environ["wsgi.input"].read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            data = json.loads(request_body) if request_body.strip() else {}
        except Exception as e:
            err = json.dumps({"error": f"Invalid JSON: {str(e)}"}).encode("utf-8")
            headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(err)))])
            start_response("400 Bad Request", headers)
            return [err]

        # Read API key
        api_key = data.get("api_key") or environ.get("HTTP_X_GROQ_API_KEY", "")
        if not api_key:
            auth = environ.get("HTTP_AUTHORIZATION", "")
            if auth.startswith("Bearer "):
                api_key = auth[7:].strip()
        api_key = api_key or os.getenv("GROQ_API_KEY", "")

        groq_service = GroqService(api_key=api_key)

        if path == "/api/test-key":
            res = groq_service.test_connection(api_key=api_key)
            status_code = "200 OK" if res.get("success") else "400 Bad Request"
            body = json.dumps(res).encode("utf-8")
            headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            start_response(status_code, headers)
            return [body]

        if path == "/api/analyze-heuristic":
            prompt = data.get("prompt", "")
            res = PromptShotDetector.analyze(prompt)
            body = json.dumps(res).encode("utf-8")
            headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            start_response("200 OK", headers)
            return [body]

        if path == "/api/analyze-groq":
            prompt = data.get("prompt", "")
            model = data.get("model", "llama-3.3-70b-versatile")
            heuristic = PromptShotDetector.analyze(prompt)
            try:
                deep = groq_service.analyze_prompt_deep(prompt, api_key=api_key, model=model)
                body = json.dumps({"heuristic": heuristic, "deep": deep}).encode("utf-8")
            except Exception as e:
                body = json.dumps({"heuristic": heuristic, "deep": None, "groq_error": str(e)}).encode("utf-8")
            headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            start_response("200 OK", headers)
            return [body]

        if path == "/api/execute":
            prompt = data.get("prompt", "")
            system_prompt = data.get("system_prompt", "")
            model = data.get("model", "llama-3.3-70b-versatile")
            temperature = float(data.get("temperature", 0.7))
            max_tokens = int(data.get("max_tokens", 1024))
            top_p = float(data.get("top_p", 1.0))

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
                body = json.dumps(res).encode("utf-8")
                headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
                start_response("200 OK", headers)
                return [body]
            except Exception as e:
                error_str = str(e)
                if "401" in error_str or "invalid_api_key" in error_str or "Invalid API Key" in error_str:
                    friendly = "Invalid API key. Double-check that you copied the full key from console.groq.com/keys."
                elif "No API key" in error_str or "Invalid key format" in error_str:
                    friendly = error_str
                elif "429" in error_str or "rate_limit" in error_str:
                    friendly = "Rate limit reached on your Groq key. Please wait a moment and try again."
                elif "404" in error_str or "does not exist" in error_str:
                    friendly = f"Model {model} was not found or accessible for this key."
                else:
                    friendly = error_str.split("\n")[0][:150]
                err = json.dumps({"error": friendly}).encode("utf-8")
                headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(err)))])
                start_response("500 Internal Server Error", headers)
                return [err]

        if path == "/api/transform":
            prompt = data.get("prompt", "")
            target_shot = data.get("target_shot", "few-shot")
            model = data.get("model", "llama-3.3-70b-versatile")

            try:
                res = groq_service.transform_prompt(prompt=prompt, target_shot=target_shot, api_key=api_key, model=model)
                body = json.dumps(res).encode("utf-8")
                headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
                start_response("200 OK", headers)
                return [body]
            except Exception as e:
                error_str = str(e)
                if "401" in error_str or "invalid_api_key" in error_str or "Invalid API Key" in error_str:
                    friendly = "Invalid API key. Double-check that you copied the full key from console.groq.com/keys."
                elif "No API key" in error_str or "Invalid key format" in error_str:
                    friendly = error_str
                elif "429" in error_str or "rate_limit" in error_str:
                    friendly = "Rate limit reached on your Groq key. Please wait a moment and try again."
                elif "404" in error_str or "does not exist" in error_str:
                    friendly = f"Model {model} was not found or accessible for this key."
                else:
                    friendly = error_str.split("\n")[0][:150]
                err = json.dumps({"error": friendly}).encode("utf-8")
                headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(err)))])
                start_response("500 Internal Server Error", headers)
                return [err]

    # 5. Static Files
    clean_path = "index.html" if path in ("/", "") else path.lstrip("/")
    file_path = os.path.join(STATIC_DIR, clean_path)

    # Protect against directory traversal
    try:
        real_file = os.path.realpath(file_path)
        real_static = os.path.realpath(STATIC_DIR)
        if not real_file.startswith(real_static):
            start_response("403 Forbidden", headers)
            return [b"Forbidden"]
    except Exception:
        start_response("404 Not Found", headers)
        return [b"Not Found"]

    if os.path.exists(file_path) and os.path.isfile(file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"
        with open(file_path, "rb") as f:
            content = f.read()
        headers.extend([
            ("Content-Type", mime_type),
            ("Content-Length", str(len(content)))
        ])
        start_response("200 OK", headers)
        return [content]

    start_response("404 Not Found", headers)
    return [b"404 Not Found"]


# Alias for frameworks expecting 'app'
app = application
