"""
PromptForge Offline — Flask application.

Serves the existing index.html and calls the local prompt-only engine.
No online GPT/Gemini API is used.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
from flask import Flask, jsonify, render_template, request

from prompt_engine import get_knowledge_base_status, process_message

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
if not (TEMPLATE_DIR / "index.html").is_file() and (BASE_DIR / "index.html").is_file():
    TEMPLATE_DIR = BASE_DIR

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TAGS_URL = OLLAMA_BASE_URL + "/api/tags"
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:3b").strip()

SUPPORTED_MODELS = {"gpt_luna", "gemini_flash"}
MAX_MESSAGE_LENGTH = 30000
MAX_HISTORY_MESSAGES = 24
MAX_HISTORY_ITEM_LENGTH = 12000

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("PromptForge.App")

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


def error_response(message: str, code: int = 400):
    return jsonify({"status": "error", "assistant_message": message}), code


def check_ollama() -> Dict[str, Any]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        response.raise_for_status()
        payload = response.json()
        names = set()
        for item in payload.get("models", []):
            if item.get("name"):
                names.add(item["name"])
            if item.get("model"):
                names.add(item["model"])
        present = LOCAL_LLM_MODEL in names or (LOCAL_LLM_MODEL + ":latest") in names
        return {
            "available": True,
            "message": "Ollama is running.",
            "configured_model": LOCAL_LLM_MODEL,
            "model_installed": present,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "available": False,
            "message": "Cannot connect to local Ollama. Start Ollama and retry.",
            "configured_model": LOCAL_LLM_MODEL,
            "model_installed": False,
        }
    except ValueError:
        return {
            "available": False,
            "message": "Ollama returned an invalid health response.",
            "configured_model": LOCAL_LLM_MODEL,
            "model_installed": False,
        }


def validate_payload(data: Any) -> Tuple[Dict[str, Any] | None, str | None]:
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return None, "Please enter a prompt or task description."
    message = message.strip()
    if len(message) > MAX_MESSAGE_LENGTH:
        return None, f"Message exceeds the {MAX_MESSAGE_LENGTH}-character limit."

    target_model = data.get("target_model")
    if target_model not in SUPPORTED_MODELS:
        return None, "Select GPT-5.6 Luna or Gemini 3.8 Flash."

    coding_mode = data.get("coding_mode")
    if type(coding_mode) is not bool:
        return None, "Coding mode must be true or false."

    history = data.get("history", [])
    if not isinstance(history, list):
        return None, "Conversation history must be a list."

    cleaned = []
    for index, item in enumerate(history[-MAX_HISTORY_MESSAGES:]):
        if not isinstance(item, dict):
            return None, f"Invalid conversation history entry {index}."
        role, content = item.get("role"), item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            return None, f"Invalid role or content in history entry {index}."
        if len(content) > MAX_HISTORY_ITEM_LENGTH:
            content = content[-MAX_HISTORY_ITEM_LENGTH:]
        cleaned.append({"role": role, "content": content})

    return {
        "message": message,
        "history": cleaned,
        "target_model": target_model,
        "coding_mode": coding_mode,
    }, None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    ollama = check_ollama()
    knowledge = get_knowledge_base_status()
    healthy = ollama["available"] and ollama["model_installed"] and knowledge["available"]
    result = {
        "status": "healthy" if healthy else "unhealthy",
        "application": {"available": True},
        "ollama": ollama,
        "knowledge_base": knowledge,
        "processing": {
            "inference_calls_per_turn": 1,
            "verification_passes": 2,
            "verification_type": "deterministic Python checks",
            "maximum_ollama_timeout_seconds": min(
                max(float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "55")), 5.0), 60.0
            ),
        },
    }
    return jsonify(result), (200 if healthy else 503)


@app.route("/api/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return error_response("Send the request as JSON.", 415)
    try:
        data = request.get_json(silent=False)
    except Exception:
        return error_response("The request contains malformed JSON.", 400)

    payload, problem = validate_payload(data)
    if problem:
        return error_response(problem, 400)

    LOGGER.info(
        "Prompt request: model=%s coding=%s chars=%s",
        payload["target_model"], payload["coding_mode"], len(payload["message"])
    )
    try:
        result = process_message(
            payload["message"],
            payload["history"],
            payload["target_model"],
            payload["coding_mode"],
        )
        if not isinstance(result, dict):
            raise ValueError("Prompt engine returned an invalid response object.")

        status = result.get("status")
        if status == "error":
            return jsonify(result), 422

        if status == "needs_clarification":
            questions = result.get("questions")
            if not isinstance(questions, list) or not questions:
                raise ValueError("Prompt engine returned an empty clarification list.")
            return jsonify(result), 200

        if status == "prompt_ready":
            if not isinstance(result.get("final_prompt"), str) or not result["final_prompt"].strip():
                raise ValueError("Prompt engine returned an empty final prompt.")
            trace = result.get("guideline_traceability")
            if not isinstance(trace, dict):
                raise ValueError("Prompt engine did not return guideline traceability.")
            return jsonify(result), 200

        raise ValueError(f"Unknown prompt-engine status: {status!r}")

    except Exception as exc:
        LOGGER.exception("Chat endpoint failed.")
        return error_response(f"Server error ({type(exc).__name__}). Check the terminal log.", 500)


@app.route("/api/knowledge/status", methods=["GET"])
def knowledge_status():
    status = get_knowledge_base_status()
    return jsonify({"status": "ready" if status["available"] else "error", "knowledge_base": status}), (
        200 if status["available"] else 503
    )


@app.errorhandler(413)
def too_large(_error):
    return error_response("The request is too large. Shorten the message or conversation history.", 413)


if __name__ == "__main__":
    LOGGER.info("Starting PromptForge Offline.")
    LOGGER.info("Configured Ollama model: %s", LOCAL_LLM_MODEL)
    LOGGER.info("Knowledge base status: %s", get_knowledge_base_status())
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
        threaded=True,
    )
