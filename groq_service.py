"""
Groq API Integration Service for LLM Playground
Provides deep prompt analysis, execution, performance measurement, and prompt transformation.
"""

import os
import json
import time
from typing import Dict, Any, Optional

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


SUPPORTED_MODELS = [
    {
        "id": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B Versatile",
        "context_window": 128000,
        "description": "Meta's flagship 70B model. Exceptional reasoning, zero-shot and few-shot understanding."
    },
    {
        "id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B Instant",
        "context_window": 128000,
        "description": "Ultra-fast response speed (over 800 tok/s), great for instant prompt classification and testing."
    },
    {
        "id": "mixtral-8x7b-32768",
        "name": "Mixtral 8x7B",
        "context_window": 32768,
        "description": "Mistral AI's sparse mixture-of-experts model, very capable in multiple languages and formatting."
    },
    {
        "id": "gemma2-9b-it",
        "name": "Gemma 2 9B IT",
        "context_window": 8192,
        "description": "Google's efficient 9B parameter instruction-tuned model."
    }
]


class GroqService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    def get_client(self, api_key: Optional[str] = None) -> "Groq":
        key = api_key or self.api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise ValueError("Groq API key is missing. Please provide it in the UI or set the GROQ_API_KEY environment variable.")
        if not GROQ_AVAILABLE:
            raise RuntimeError("The 'groq' package is not installed in the python environment.")
        return Groq(api_key=key)

    def test_connection(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Test if the provided Groq API key is valid by querying available models or a small completion."""
        try:
            client = self.get_client(api_key)
            start_time = time.time()
            # Fast test completion
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0.0
            )
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "success": True,
                "latency_ms": elapsed_ms,
                "message": "Groq API connection successful!"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def analyze_prompt_deep(self, prompt: str, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
        """
        Use Groq to perform a deep semantic analysis of the prompt,
        classifying whether it's zero-shot, one-shot, or few-shot,
        extracting demonstration pairs, and identifying prompt engineering techniques.
        """
        client = self.get_client(api_key)

        system_instruction = (
            "You are an expert Prompt Engineering Diagnostics Engine. "
            "Your job is to analyze any given user prompt and categorize whether it is: "
            "1. 'zero-shot' (no demonstrations/input-output examples provided, directly asking the task) "
            "2. 'one-shot' (exactly 1 demonstration/input-output example provided before the actual task) "
            "3. 'few-shot' (2 or more demonstration/input-output examples provided). "
            "\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "classification": "zero-shot" | "one-shot" | "few-shot",\n'
            '  "example_count": number,\n'
            '  "confidence": number (between 0.0 and 1.0),\n'
            '  "explanation": "concise explanation of why this classification was assigned",\n'
            '  "extracted_examples": [\n'
            '     {"example_num": 1, "input": "...", "output": "..."}\n'
            '  ],\n'
            '  "target_task": "the final query, problem, or text the prompt actually wants solved",\n'
            '  "techniques_detected": ["Zero-shot instruction", "Few-shot in-context learning", "Chain of thought", "Role-based system prompt", "Structured delimiters", "Negative constraints"],\n'
            '  "prompt_quality_score": number (1-100),\n'
            '  "strengths": ["list of what the prompt does well"],\n'
            '  "improvement_suggestions": ["actionable recommendations to improve output accuracy"]\n'
            "}"
        )

        user_content = f"Analyze this prompt:\n```\n{prompt}\n```"

        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500
        )
        elapsed_sec = time.time() - start_time

        raw_content = completion.choices[0].message.content
        usage = completion.usage

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            # Fallback parsing
            clean_str = raw_content.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            parsed = json.loads(clean_str.strip())

        parsed["meta"] = {
            "model_used": model,
            "latency_ms": round(elapsed_sec * 1000, 2),
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "tokens_per_second": round(usage.completion_tokens / elapsed_sec, 2) if usage and elapsed_sec > 0 else 0
        }

        return parsed

    def execute_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the prompt on Groq and measure inference speed and tokens.
        """
        client = self.get_client(api_key)

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        elapsed_sec = time.time() - start_time

        content = completion.choices[0].message.content
        usage = completion.usage

        tok_per_sec = 0.0
        if usage and usage.completion_tokens and elapsed_sec > 0:
            tok_per_sec = round(usage.completion_tokens / elapsed_sec, 2)

        return {
            "output": content,
            "model": model,
            "finish_reason": completion.choices[0].finish_reason,
            "metrics": {
                "latency_ms": round(elapsed_sec * 1000, 2),
                "tokens_per_second": tok_per_sec,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
        }

    def transform_prompt(
        self,
        prompt: str,
        target_shot: str,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile"
    ) -> Dict[str, Any]:
        """
        Convert a prompt into zero-shot, one-shot, or few-shot format.
        """
        client = self.get_client(api_key)

        sys_msg = (
            f"You are an expert prompt engineer. The user has an existing prompt and wants you to rewrite it "
            f"into a '{target_shot}' prompt format.\n"
            f"- If target is 'zero-shot': Provide clear, direct task instructions without any solved examples.\n"
            f"- If target is 'one-shot': Provide clear instructions, followed by exactly 1 high-quality, solved demonstration example, followed by the target placeholder.\n"
            f"- If target is 'few-shot': Provide clear instructions, followed by 3-4 diverse, high-quality demonstration examples, followed by the target placeholder.\n"
            f"Output JSON with keys: 'transformed_prompt', 'explanation_of_changes', 'added_examples_count'."
        )

        user_msg = f"Original prompt:\n```\n{prompt}\n```\nTarget shot style: {target_shot}"

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500
        )

        raw = completion.choices[0].message.content
        return json.loads(raw)
