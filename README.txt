# PromptForge Offline — Fast Release

## Files
- app.py
- prompt_engine.py
- knowledge_base.json
- index.html

## Install dependencies
```powershell
py -m pip install flask requests
```

## Start Ollama and install the recommended lightweight model
```powershell
ollama pull qwen2.5:3b
```

## Run PromptForge (PowerShell)
```powershell
$env:LOCAL_LLM_MODEL="qwen2.5:3b"
$env:OLLAMA_TIMEOUT_SECONDS="55"
$env:OLLAMA_NUM_CTX="4096"
$env:OLLAMA_NUM_PREDICT="900"
python app.py
```

Open http://127.0.0.1:5000/

## Processing design
- One local Ollama generation request per turn.
- Two deterministic Python verification passes.
- Permanent JSON knowledge base; PDFs are not read at runtime.
- Strict model-specific guideline filtering.
- Coding rules are included only when Coding Mode is ON and task classification identifies software development.
- The engine never answers the user's underlying task; it only creates an instruction prompt or asks clarification questions.

## Notes
The 55-second timeout applies to the Ollama request. Actual completion time depends on CPU speed, prompt size, model warm-up and RAM. This is not a guarantee that every request will finish within 60 seconds.
