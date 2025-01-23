import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ichigo.llm.ichigo import IchigoAssistant

app = FastAPI(
    title="Ichigo LLM API",
    description="API for audio processing using Ichigo LLM",
    version="0.0.1",
)

# Initialize assistants
ichigo_assistant = None
speechless_assistant = None


def get_assistant(use_speechless=False):
    global ichigo_assistant, speechless_assistant
    if use_speechless:
        if speechless_assistant is None:
            speechless_assistant = IchigoAssistant(use_speechless=True)
        return speechless_assistant
    else:
        if ichigo_assistant is None:
            ichigo_assistant = IchigoAssistant(use_speechless=False)
        return ichigo_assistant


@app.post("/chat/")
async def process_audio_file(
    file: UploadFile = File(...),
    max_new_tokens: int = 2048,
    use_speechless: bool = False,
):
    """
    Process an audio file uploaded via HTTP POST request

    Args:
        file: Audio file to process
        max_new_tokens: Maximum number of tokens to generate
        use_speechless: Whether to use the Speechless model instead of Ichigo
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(400, "Unsupported file format")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        assistant = get_assistant(use_speechless)
        response = assistant.generate_text(tmp_path, max_new_tokens=max_new_tokens)
        return JSONResponse(content={"response": response})

    except Exception as e:
        raise HTTPException(500, str(e))

    finally:
        os.unlink(tmp_path)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Ichigo LLM API. Send POST requests to /chat/ endpoint."
    }
