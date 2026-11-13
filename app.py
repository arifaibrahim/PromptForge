"""
PromptForge Offline — Flask application entry point.

Responsibilities:
- Serve the existing templates/index.html interface.
- Validate chat requests.
- Check local Ollama and the permanent knowledge_base.json.
- Call prompt_engine.process_message().
- Return the response contract expected by the existing frontend.

No online GPT/Gemini APIs are used. Ollama is accessed only on localhost.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request

from prompt_engine import (
    get_knowledge_base_status,
    process_message,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
if not (TEMPLATE_DIR / "index.html").is_file() and (BASE_DIR / "index.html").is_file():
    TEMPLATE_DIR = BASE_DIR

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "").strip()

SUPPORTED_MODELS = {"gpt_luna", "gemini_flash"}
SUPPORTED_ROLES = {"user", "assistant"}

MAX_MESSAGE_LENGTH = 50000
MAX_HISTORY_MESSAGES = 40
MAX_HISTORY_CONTENT_LENGTH = 30000

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("PromptForge.App")

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


def _error(message: str, status_code: int = 400):
    return jsonify({
        "status": "error",
        "assistant_message": str(message),
    }), status_code


def _check_ollama() -> dict[str, Any]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        return {"available": True, "message": "Ollama is running."}
    except requests.exceptions.ConnectionError:
        return {
            "available": False,
            "message": "Cannot connect to Ollama. Start the local Ollama service.",
        }
    except requests.exceptions.Timeout:
        return {"available": False, "message": "Ollama health check timed out."}
    except requests.exceptions.RequestException as exc:
        LOGGER.warning("Ollama health check failed: %s", exc)
        return {"available": False, "message": "Ollama service is unavailable."}


def _check_local_model() -> dict[str, Any]:
    if not LOCAL_LLM_MODEL:
        return {
            "available": False,
            "model": None,
            "message": "LOCAL_LLM_MODEL is not configured.",
        }

    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        installed = set()
        for item in data.get("models", []):
            if item.get("name"):
                installed.add(item["name"])
            if item.get("model"):
                installed.add(item["model"])

        configured = LOCAL_LLM_MODEL
        alternatives = {configured, f"{configured}:latest"}
        available = bool(installed.intersection(alternatives))
        return {
            "available": available,
            "model": configured,
            "message": (
                "Configured local model is installed."
                if available
                else f"Model '{configured}' is not installed in Ollama."
            ),
        }
    except (requests.exceptions.RequestException, ValueError) as exc:
        LOGGER.warning("Could not verify local Ollama model: %s", exc)
        return {
            "available": False,
            "model": LOCAL_LLM_MODEL,
            "message": "Unable to verify the configured local model.",
        }


def _validate_chat_payload(data: Any):
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return None, "Please enter a message before sending."
    message = message.strip()
    if len(message) > MAX_MESSAGE_LENGTH:
        return None, f"Message exceeds the {MAX_MESSAGE_LENGTH}-character limit."

    target_model = data.get("target_model")
    if target_model not in SUPPORTED_MODELS:
        return None, "Select either GPT-5.6 Luna or Gemini 3.8 Flash."

    coding_mode = data.get("coding_mode")
    if type(coding_mode) is not bool:
        return None, "coding_mode must be true or false."

    history = data.get("history", [])
    if not isinstance(history, list):
        return None, "Conversation history must be a list."
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    clean_history = []
    for index, item in enumerate(history):
        if not isinstance(item, dict):
            return None, f"Invalid history entry at position {index}."
        role = item.get("role")
        content = item.get("content")
        if role not in SUPPORTED_ROLES or not isinstance(content, str):
            return None, f"Invalid role or content in history entry {index}."
        if len(content) > MAX_HISTORY_CONTENT_LENGTH:
            return None, f"History entry {index} is too long."
        clean_history.append({"role": role, "content": content})

    return {
        "message": message,
        "history": clean_history,
        "target_model": target_model,
        "coding_mode": coding_mode,
    }, None


def _validate_engine_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Prompt engine returned a non-object response.")

    status = result.get("status")
    if status == "error":
        message = result.get("assistant_message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Prompt engine returned an invalid error response.")
        return {"status": "error", "assistant_message": message}

    common = {
        "assumptions": result.get("assumptions", []),
        "applied_guidelines": result.get("applied_guidelines", []),
        "suggested_settings": result.get("suggested_settings"),
        "guideline_traceability": result.get("guideline_traceability"),
    }

    if status == "needs_clarification":
        assistant_message = result.get("assistant_message")
        questions = result.get("questions")
        if not isinstance(assistant_message, str) or not assistant_message.strip():
            raise ValueError("Clarification response has no assistant message.")
        if not isinstance(questions, list) or not questions:
            raise ValueError("Clarification response has no questions.")
        if not all(isinstance(question, str) and question.strip() for question in questions):
            raise ValueError("Clarification response contains an invalid question.")
        return {
            "status": "needs_clarification",
            "assistant_message": assistant_message,
            "questions": questions,
            "final_prompt": None,
            **common,
        }

    if status == "prompt_ready":
        assistant_message = result.get("assistant_message")
        final_prompt = result.get("final_prompt")
        if not isinstance(assistant_message, str) or not assistant_message.strip():
            raise ValueError("Prompt-ready response has no assistant message.")
        if not isinstance(final_prompt, str) or not final_prompt.strip():
            raise ValueError("Prompt-ready response has an empty final prompt.")
        if not isinstance(common["applied_guidelines"], list):
            raise ValueError("Applied guideline identifiers must be a list.")
        trace = common["guideline_traceability"]
        if not isinstance(trace, dict):
            raise ValueError("Prompt-ready response has no guideline traceability report.")
        if trace.get("coverage_status") != "complete_for_retrieval_policy_and_both_verification_passes":
            raise ValueError("Guideline traceability did not confirm both verification passes.")
        return {
            "status": "prompt_ready",
            "assistant_message": assistant_message,
            "questions": [],
            "final_prompt": final_prompt,
            **common,
        }

    raise ValueError(f"Unknown prompt-engine status: {status!r}")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    ollama = _check_ollama()
    model = _check_local_model() if ollama["available"] else {
        "available": False,
        "model": LOCAL_LLM_MODEL or None,
        "message": "Cannot verify the model because Ollama is unavailable.",
    }
    knowledge = get_knowledge_base_status()

    healthy = ollama["available"] and model["available"] and knowledge["available"]
    payload = {
        "status": "healthy" if healthy else "unhealthy",
        "application": {
            "available": True,
            "message": "PromptForge Flask application is running.",
        },
        "ollama": ollama,
        "local_model": model,
        "knowledge_base": knowledge,
    }
    return jsonify(payload), (200 if healthy else 503)


@app.route("/api/knowledge/status", methods=["GET"])
def knowledge_status():
    """Return knowledge-base counts and validation state for diagnostics."""
    status = get_knowledge_base_status()
    return jsonify({
        "status": "ready" if status["available"] else "error",
        "knowledge_base": status,
    }), (200 if status["available"] else 503)


@app.route("/api/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return _error("Send the chat request as JSON.", 415)

    try:
        payload = request.get_json(silent=False)
    except Exception:
        return _error("Malformed JSON request.", 400)

    validated, validation_error = _validate_chat_payload(payload)
    if validation_error:
        return _error(validation_error, 400)

    LOGGER.info(
        "Chat request | target_model=%s | coding_mode=%s | message_chars=%s",
        validated["target_model"],
        validated["coding_mode"],
        len(validated["message"]),
    )

    try:
        result = process_message(
            validated["message"],
            validated["history"],
            validated["target_model"],
            validated["coding_mode"],
        )
        response = _validate_engine_result(result)
        return jsonify(response), (200 if response["status"] != "error" else 422)

    except requests.exceptions.Timeout:
        LOGGER.warning("Request to local Ollama timed out.")
        return _error("The local AI model timed out. Increase OLLAMA_TIMEOUT_SECONDS or use a smaller model.", 504)
    except requests.exceptions.ConnectionError:
        LOGGER.warning("Cannot connect to Ollama.")
        return _error("Cannot connect to Ollama. Start the local Ollama service.", 503)
    except ValueError as exc:
        LOGGER.exception("Invalid engine response.")
        return _error(f"Prompt engine response validation failed: {exc}", 500)
    except Exception as exc:
        LOGGER.exception("Unexpected chat route error.")
        return _error(f"Unexpected server error ({type(exc).__name__}). Check the terminal log.", 500)


@app.errorhandler(413)
def request_too_large(_error_value):
    return _error("The request is too large. Shorten the message or conversation history.", 413)


if __name__ == "__main__":
    LOGGER.info("Starting PromptForge Offline.")
    LOGGER.info("Knowledge base: %s", get_knowledge_base_status())
    LOGGER.info("Configured local Ollama model: %s", LOCAL_LLM_MODEL or "(not set)")
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
        threaded=True,
    )
