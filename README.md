# Ichigo Package

## PyPI

<!-- 
python -m build
pip install dist/ichigo-0.0.1-py3-none-any.whl
python -c "import ichigo.asr as asr; print(asr.__file__)" 
python -c "from ichigo.asr import transcribe; results = transcribe('speech.wav'); print(results)"
python -c "from ichigo.asr import transcribe; results = transcribe('speech.wav', return_stoks=True); print(results)"
python -c "from ichigo.asr import transcribe; results = transcribe('/root/ichigo-experiments/test'); print(results)"
python -c "from ichigo.llm import process_audio; response = process_audio('speech.wav'); print(response)"
-->

1. Setup package with `python=3.10` (dev)

```
python -m build
python -m twine upload dist/* 
```

2. Install python package

```bash
pip install ichigo
```

## ASR

### Batch Processing

Transcribe with your audio file

```python
# Quick one-liner
from ichigo.asr import transcribe
results = transcribe("path/to/file/or/folder")

# Or with more control using the model class
from ichigo.asr import IchigoASR
model = IchigoASR(config="config-name")
results = model.transcribe(
    "path/to/file/or/folder",
    output_path="./output_folder",
    extensions=(".wav", ".mp3", ".flac", ".m4a")
)
```

### API

```bash
# Start the API server
uvicorn ichigo.asr.server:app --host 0.0.0.0 --port 8000

# Use with curl
curl -X POST "http://localhost:8000/transcribe/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"

# Get semantic tokens
curl -X POST "http://localhost:8000/transcribe/?return_stoks=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"
```

You can also access the API documentation at `http://localhost:8000/docs`


## LLMs

### Python Usage

```python
# Quick one-liner for audio processing
from ichigo.llm import process_audio
response = process_audio("path/to/audio.wav")

# Or with more control using the assistant class
from ichigo.llm import IchigoAssistant
# Use default Ichigo model
assistant = IchigoAssistant()
# Or use Speechless model
assistant = IchigoAssistant(use_speechless=True)

response = assistant.generate_text(
    "path/to/audio.wav",
    max_new_tokens=2048
)
```

### API

```bash
# Start the API server
uvicorn ichigo.llm.server:app --host 0.0.0.0 --port 8001

# Use with default Ichigo model
curl -X POST "http://localhost:8001/chat/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"

# Use with Speechless model
curl -X POST "http://localhost:8001/chat/?use_speechless=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"
```

You can also access the API documentation at `http://localhost:8001/docs`