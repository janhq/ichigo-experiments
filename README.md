# Ichigo Package

## PyPI

<!-- 
python -m build
pip install dist/ichigo-0.0.1-py3-none-any.whl
python -c "import ichigo.asr as asr; print(asr.__file__)" 
python -c "from ichigo.asr import transcribe; results = transcribe('speech.wav'); print(results)"
python -c "from ichigo.asr import get_stoks; stoks = get_stoks('speech.wav'); print(stoks)"
python -c "from ichigo.asr import transcribe; results = transcribe('/root/ichigo-experiments/test'); print(results)"
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
# Quick one-liner for transcription
from ichigo.asr import transcribe, get_stoks
results = transcribe("path/to/file/or/folder")
tokens = get_stoks("path/to/file")

# Or with more control using the model class
from ichigo.asr import IchigoASR
model = IchigoASR(config="config-name")
results = model.transcribe(
    "path/to/file/or/folder",
    output_path="./output_folder",
    extensions=(".wav", ".mp3", ".flac", ".m4a")
)
stoks = model.get_stoks("path/to/file")
```

### API

```bash
# Start the API server
uvicorn ichigo.asr.server:app --host 0.0.0.0 --port 8000

# Use with curl for transcription
curl -X POST "http://localhost:8000/transcribe/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"

# Get semantic tokens
curl -X POST "http://localhost:8000/get_stoks/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/audio.wav"
```

You can also access the API documentation at `http://localhost:8000/docs`