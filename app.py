"""
PromptForge Offline
AI-Powered Prompt Engineering and Optimization Assistant

Main responsibilities:
1. Initialize Flask.
2. Serve the frontend.
3. Check local Ollama and PDF availability.
4. Validate incoming chat requests.
5. Communicate with prompt_engine.py.
6. Return standardized JSON responses.
7. Handle errors safely.

All AI inference is performed locally through Ollama.
"""

import os
import logging
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader

from prompt_engine import process_message


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

LOCAL_LLM_MODEL = os.environ.get(
    "LOCAL_LLM_MODEL",
    "llama3.2"
)

OLLAMA_TIMEOUT = int(
    os.environ.get("OLLAMA_TIMEOUT", "120")
)

# Expected knowledge-base documents and page counts.
REQUIRED_PDFS = {
    "general_guidelines.pdf": 33,
    "gpt_luna_guidelines.pdf": 47,
    "gemini_flash_guidelines.pdf": 43,
    "coding_guidelines.pdf": 40
}

# Supported target models.
SUPPORTED_MODELS = {
    "gpt_luna",
    "gemini_flash"
}

# Supported conversation roles.
SUPPORTED_ROLES = {
    "user",
    "assistant"
}

# Maximum accepted request sizes.
MAX_MESSAGE_LENGTH = 50000
MAX_HISTORY_MESSAGES = 100
MAX_HISTORY_CONTENT_LENGTH = 50000

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("PromptForge")


# ============================================================
# 2. FLASK INITIALIZATION
# ============================================================

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates")
)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


# ============================================================
# 3. STANDARDIZED ERROR RESPONSE
# ============================================================

def error_response(message, status_code=400):
    """
    Returns the standardized error response.
    """

    return jsonify({
        "status": "error",
        "assistant_message": str(message)
    }), status_code


# ============================================================
# 4. HEALTH CHECK HELPERS
# ============================================================

def check_ollama_connection():
    """
    Checks whether the local Ollama service is running.

    Returns:
        dict: Connection status and information.
    """

    try:
        response = requests.get(
            OLLAMA_TAGS_URL,
            timeout=5
        )

        response.raise_for_status()

        return {
            "available": True,
            "message": "Ollama is running."
        }

    except requests.exceptions.ConnectionError:

        return {
            "available": False,
            "message": (
                "Cannot connect to Ollama. "
                "Start the local Ollama service."
            )
        }

    except requests.exceptions.Timeout:

        return {
            "available": False,
            "message": "Ollama connection timed out."
        }

    except requests.exceptions.RequestException as exc:

        logger.error("Ollama health check failed: %s", exc)

        return {
            "available": False,
            "message": "Ollama service is unavailable."
        }


def check_local_model():
    """
    Checks whether the configured local model
    is installed in Ollama.

    Returns:
        dict: Model availability information.
    """

    if not LOCAL_LLM_MODEL.strip():

        return {
            "available": False,
            "model": None,
            "message": "LOCAL_LLM_MODEL is not configured."
        }

    try:

        response = requests.get(
            OLLAMA_TAGS_URL,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        installed_models = data.get("models", [])

        installed_names = [
            model.get("name", "")
            for model in installed_models
        ]

        installed_names.extend(
            model.get("model", "")
            for model in installed_models
        )

        configured_name = LOCAL_LLM_MODEL.strip()

        # Match exact names or the implicit :latest tag.
        possible_names = {
            configured_name,
            f"{configured_name}:latest"
        }

        model_available = any(
            name in possible_names
            for name in installed_names
        )

        if model_available:

            return {
                "available": True,
                "model": configured_name,
                "message": "Configured local model is installed."
            }

        return {
            "available": False,
            "model": configured_name,
            "message": (
                f"Model '{configured_name}' is not installed "
                "in Ollama."
            )
        }

    except (
        requests.exceptions.RequestException,
        ValueError
    ) as exc:

        logger.error("Local model check failed: %s", exc)

        return {
            "available": False,
            "model": LOCAL_LLM_MODEL,
            "message": "Unable to verify the local model."
        }


def check_knowledge_base():
    """
    Checks all four required PDF documents.

    Validates:
    - File availability.
    - PDF readability.
    - Expected page count.

    Does not modify the original PDFs.

    Returns:
        dict: Knowledge-base validation results.
    """

    results = {}
    all_available = True

    for filename, expected_pages in REQUIRED_PDFS.items():

        pdf_path = KNOWLEDGE_DIR / filename

        if not pdf_path.exists():

            results[filename] = {
                "available": False,
                "readable": False,
                "expected_pages": expected_pages,
                "actual_pages": 0,
                "message": "PDF file is missing."
            }

            all_available = False
            continue

        try:

            if not pdf_path.is_file():

                raise ValueError("Path is not a regular file.")

            reader = PdfReader(
                str(pdf_path),
                strict=False
            )

            actual_pages = len(reader.pages)

            # Attempt to access each page's extracted text.
            extracted_pages = 0

            for page in reader.pages:

                page.extract_text()

                extracted_pages += 1

            readable = (
                actual_pages > 0
                and extracted_pages == actual_pages
            )

            correct_page_count = (
                actual_pages == expected_pages
            )

            valid = readable and correct_page_count

            if not readable:

                message = "PDF content could not be fully read."

            elif not correct_page_count:

                message = (
                    f"Expected {expected_pages} pages, "
                    f"but found {actual_pages} pages."
                )

            else:

                message = "PDF is available and readable."

            results[filename] = {
                "available": True,
                "readable": readable,
                "expected_pages": expected_pages,
                "actual_pages": actual_pages,
                "message": message
            }

            if not valid:
                all_available = False

        except Exception as exc:

            logger.error(
                "PDF validation failed for %s: %s",
                filename,
                exc
            )

            results[filename] = {
                "available": True,
                "readable": False,
                "expected_pages": expected_pages,
                "actual_pages": 0,
                "message": "PDF validation or extraction failed."
            }

            all_available = False

    return {
        "available": all_available,
        "documents": results
    }


# ============================================================
# 5. RESPONSE CONTRACT VALIDATION
# ============================================================

def validate_engine_response(result):
    """
    Validates the dictionary returned by prompt_engine.py.

    Prevents malformed or incomplete responses from being
    presented as successfully generated prompts.
    """

    if not isinstance(result, dict):

        raise ValueError(
            "Prompt engine returned an invalid response."
        )

    status = result.get("status")

    # --------------------------------------------------------
    # ERROR RESPONSE
    # --------------------------------------------------------

    if status == "error":

        assistant_message = result.get("assistant_message")

        if not isinstance(assistant_message, str):

            raise ValueError(
                "Prompt engine returned an invalid error response."
            )

        return {
            "status": "error",
            "assistant_message": assistant_message
        }

    # --------------------------------------------------------
    # CLARIFICATION RESPONSE
    # --------------------------------------------------------

    if status == "needs_clarification":

        assistant_message = result.get("assistant_message")
        questions = result.get("questions")

        if not isinstance(assistant_message, str) or not assistant_message.strip():

            raise ValueError(
                "Clarification response is missing its message."
            )

        if not isinstance(questions, list):

            raise ValueError(
                "Clarification questions must be a list."
            )

        if not all(
            isinstance(question, str) and question.strip()
            for question in questions
        ):

            raise ValueError(
                "Clarification response contains invalid questions."
            )

        return {
            "status": "needs_clarification",
            "assistant_message": assistant_message,
            "questions": questions,
            "final_prompt": None,
            "assumptions": result.get("assumptions", []),
            "applied_guidelines": result.get(
                "applied_guidelines", []
            ),
            "suggested_settings": result.get(
                "suggested_settings"
            )
        }

    # --------------------------------------------------------
    # FINAL PROMPT RESPONSE
    # --------------------------------------------------------

    if status == "prompt_ready":

        assistant_message = result.get("assistant_message")
        final_prompt = result.get("final_prompt")

        if not isinstance(assistant_message, str) or not assistant_message.strip():

            raise ValueError(
                "Final response is missing its assistant message."
            )

        if not isinstance(final_prompt, str) or not final_prompt.strip():

            raise ValueError(
                "Final prompt is empty or invalid."
            )

        return {
            "status": "prompt_ready",
            "assistant_message": assistant_message,
            "questions": [],
            "final_prompt": final_prompt,
            "assumptions": result.get("assumptions", []),
            "applied_guidelines": result.get(
                "applied_guidelines", []
            ),
            "suggested_settings": result.get(
                "suggested_settings"
            )
        }

    # --------------------------------------------------------
    # UNKNOWN RESPONSE
    # --------------------------------------------------------

    raise ValueError(
        f"Unknown prompt engine response status: {status}"
    )


# ============================================================
# 6. FRONTEND ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """
    Serves the main chatbot interface.
    """

    return render_template("index.html")


# ============================================================
# 7. HEALTH CHECK ROUTE
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    """
    Checks the complete local application environment.

    Checks:
    - Flask application.
    - Ollama connection.
    - Configured local model.
    - Four required PDF documents.
    """

    ollama_status = check_ollama_connection()

    if ollama_status["available"]:
        model_status = check_local_model()
    else:
        model_status = {
            "available": False,
            "model": LOCAL_LLM_MODEL,
            "message": "Cannot verify model because Ollama is unavailable."
        }

    knowledge_status = check_knowledge_base()

    ollama_available = ollama_status["available"]

    model_available = model_status["available"]

    knowledge_available = knowledge_status["available"]

    application_healthy = (
        ollama_available
        and model_available
        and knowledge_available
    )

    response_data = {
        "status": (
            "healthy"
            if application_healthy
            else "unhealthy"
        ),

        "application": {
            "available": True,
            "message": "Flask application is running."
        },

        "ollama": ollama_status,

        "local_model": model_status,

        "knowledge_base": knowledge_status
    }

    return jsonify(response_data), (
        200 if application_healthy else 503
    )


# ============================================================
# 8. REQUEST VALIDATION
# ============================================================

def validate_chat_request(data):
    """
    Validates incoming POST /api/chat JSON data.

    Expected structure:

    {
        "message": "...",
        "history": [],
        "target_model": "gpt_luna",
        "coding_mode": false
    }

    Returns:
        tuple: (validated_data, error_message)
    """

    if not isinstance(data, dict):

        return None, "Request body must be a JSON object."

    # --------------------------------------------------------
    # MESSAGE VALIDATION
    # --------------------------------------------------------

    message = data.get("message")

    if not isinstance(message, str):

        return None, "Message must be a string."

    message = message.strip()

    if not message:

        return None, "Please enter a message before sending."

    if len(message) > MAX_MESSAGE_LENGTH:

        return None, (
            f"Message exceeds the maximum allowed length "
            f"of {MAX_MESSAGE_LENGTH} characters."
        )

    # --------------------------------------------------------
    # TARGET MODEL VALIDATION
    # --------------------------------------------------------

    target_model = data.get("target_model")

    if not isinstance(target_model, str):

        return None, "Target model must be a string."

    if target_model not in SUPPORTED_MODELS:

        return None, (
            "Invalid target model. Select either "
            "'gpt_luna' or 'gemini_flash'."
        )

    # --------------------------------------------------------
    # CODING MODE VALIDATION
    # --------------------------------------------------------

    coding_mode = data.get("coding_mode")

    # Strict Boolean validation.
    # Prevents accepting strings such as "true" or integers.
    if type(coding_mode) is not bool:

        return None, (
            "coding_mode must be a Boolean value: "
            "true or false."
        )

    # --------------------------------------------------------
    # HISTORY VALIDATION
    # --------------------------------------------------------

    history = data.get("history", [])

    if not isinstance(history, list):

        return None, "Conversation history must be a list."

    if len(history) > MAX_HISTORY_MESSAGES:

        return None, (
            f"Conversation history cannot exceed "
            f"{MAX_HISTORY_MESSAGES} messages."
        )

    validated_history = []

    for index, item in enumerate(history):

        if not isinstance(item, dict):

            return None, (
                f"Invalid conversation history entry at "
                f"position {index}."
            )

        role = item.get("role")
        content = item.get("content")

        if role not in SUPPORTED_ROLES:

            return None, (
                f"Invalid role in conversation history "
                f"at position {index}."
            )

        if not isinstance(content, str):

            return None, (
                f"Invalid message content in conversation "
                f"history at position {index}."
            )

        if len(content) > MAX_HISTORY_CONTENT_LENGTH:

            return None, (
                f"Conversation history message at position "
                f"{index} exceeds the maximum length."
            )

        validated_history.append({
            "role": role,
            "content": content
        })

    # --------------------------------------------------------
    # RETURN VALIDATED REQUEST
    # --------------------------------------------------------

    return {
        "message": message,
        "history": validated_history,
        "target_model": target_model,
        "coding_mode": coding_mode
    }, None


# ============================================================
# 9. CHAT API ROUTE
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Receives user messages and generates refined prompts
    through prompt_engine.py.

    Prompt-engineering logic is intentionally not implemented
    inside this file.
    """

    # --------------------------------------------------------
    # CHECK CONTENT TYPE
    # --------------------------------------------------------

    if not request.is_json:

        return error_response(
            "Invalid request format. Please send JSON data.",
            415
        )

    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    try:

        data = request.get_json(silent=False)

    except Exception:

        return error_response(
            "Malformed JSON request. Please check your data.",
            400
        )

    if data is None:

        return error_response(
            "Request body cannot be empty.",
            400
        )

    # --------------------------------------------------------
    # VALIDATE REQUEST
    # --------------------------------------------------------

    validated_data, validation_error = validate_chat_request(
        data
    )

    if validation_error:

        return error_response(
            validation_error,
            400
        )

    message = validated_data["message"]

    history = validated_data["history"]

    target_model = validated_data["target_model"]

    coding_mode = validated_data["coding_mode"]

    logger.info(
        "Processing prompt request | Model: %s | Coding: %s",
        target_model,
        coding_mode
    )

    # --------------------------------------------------------
    # CALL PROMPT ENGINE
    # --------------------------------------------------------

    try:

        result = process_message(
            message,
            history,
            target_model,
            coding_mode
        )

        # ----------------------------------------------------
        # VALIDATE ENGINE RESPONSE
        # ----------------------------------------------------

        validated_response = validate_engine_response(
            result
        )

        # ----------------------------------------------------
        # RETURN STANDARDIZED RESPONSE
        # ----------------------------------------------------

        return jsonify(validated_response), 200

    # --------------------------------------------------------
    # TIMEOUT ERRORS
    # --------------------------------------------------------

    except requests.exceptions.Timeout:

        logger.error(
            "Local AI request timed out."
        )

        return error_response(
            "The local AI model took too long to respond. "
            "Try again or increase the configured timeout.",
            504
        )

    # --------------------------------------------------------
    # CONNECTION ERRORS
    # --------------------------------------------------------

    except requests.exceptions.ConnectionError:

        logger.error(
            "Cannot connect to the local AI service."
        )

        return error_response(
            "Cannot connect to Ollama. Make sure the local "
            "Ollama service is running.",
            503
        )

    # --------------------------------------------------------
    # OTHER REQUEST ERRORS
    # --------------------------------------------------------

    except requests.exceptions.RequestException as exc:

        logger.error(
            "Local AI communication error: %s",
            exc
        )

        return error_response(
            "An error occurred while communicating with "
            "the local AI service.",
            502
        )

    # --------------------------------------------------------
    # INVALID ENGINE RESPONSE
    # --------------------------------------------------------

    except ValueError as exc:

        logger.error(
            "Invalid prompt engine response: %s",
            exc
        )

        return error_response(
            "The prompt engine returned an invalid or "
            "incomplete response. Please try again.",
            500
        )

    # --------------------------------------------------------
    # UNEXPECTED ERRORS
    # --------------------------------------------------------

    except Exception:

        logger.exception(
            "Unexpected error during prompt processing."
        )

        return error_response(
            "An unexpected server error occurred while "
            "processing your request.",
            500
        )


# ============================================================
# 10. GLOBAL ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return error_response(
        "The requested endpoint was not found.",
        404
    )


@app.errorhandler(405)
def method_not_allowed(error):

    return error_response(
        "The requested HTTP method is not supported.",
        405
    )


@app.errorhandler(413)
def request_too_large(error):

    return error_response(
        "Request is too large. Please reduce the message size.",
        413
    )


@app.errorhandler(500)
def internal_server_error(error):

    logger.error(
        "Internal server error: %s",
        error
    )

    return error_response(
        "An internal server error occurred.",
        500
    )


# ============================================================
# 11. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    logger.info("=" * 55)
    logger.info("Starting PromptForge Offline")
    logger.info("=" * 55)

    logger.info(
        "Local AI model: %s",
        LOCAL_LLM_MODEL
    )

    logger.info(
        "Ollama endpoint: %s",
        OLLAMA_CHAT_URL
    )

    logger.info(
        "Knowledge directory: %s",
        KNOWLEDGE_DIR
    )

    logger.info(
        "Application address: http://127.0.0.1:5000"
    )

    logger.info("=" * 55)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )