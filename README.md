# Ichigo Package

## Installation

```bash
pip install ichigo
```

## ASR

### Batch Processing

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
uvicorn app_asr:app --host 0.0.0.0 --port 8000

# alternatively, with Docker
# docker compose -f 'docker-compose.yml' up -d --build 'asr'

# Use with curl for transcription
curl "http://localhost:8000/v1/audio/transcriptions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.m4a" -F "model=ichigo"

# Get semantic tokens
curl "http://localhost:8000/s2r" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.m4a"

curl "http://192.168.100.111:8000/r2t" -X POST \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  --data '{"tokens":"<|sound_start|><|sound_1012|><|sound_end|>"}'

curl "http://192.168.100.111:8000/r2t" -X POST \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  --data '{"tokens":"<|sound_start|><|sound_1012|><|sound_1508|><|sound_1508|><|sound_0636|><|sound_1090|><|sound_0567|><|sound_0901|><|sound_0901|><|sound_1192|><|sound_1820|><|sound_0547|><|sound_1999|><|sound_0157|><|sound_0157|><|sound_1454|><|sound_1223|><|sound_1223|><|sound_1223|><|sound_1223|><|sound_1808|><|sound_1808|><|sound_1573|><|sound_0065|><|sound_1508|><|sound_1508|><|sound_1268|><|sound_0568|><|sound_1745|><|sound_1508|><|sound_0084|><|sound_1768|><|sound_0192|><|sound_1048|><|sound_0826|><|sound_0192|><|sound_0517|><|sound_0192|><|sound_0826|><|sound_0971|><|sound_1845|><|sound_1694|><|sound_1048|><|sound_0192|><|sound_1048|><|sound_1268|><|sound_end|>"}'
```

You can also access the API documentation at `http://localhost:8000/docs`
