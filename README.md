# pip install ichigo

## PyPI

<!-- 
python -m build
pip install dist/ichigo-0.0.1-py3-none-any.whl
python -c "import ichigo.asr as asr; print(asr.__file__)" 
python -c "from ichigo.asr import transcribe; text = transcribe('speech.wav', 'transcript.txt'); print(text)"
-->

1. Setup package with `python=3.10`

```
python -m build
python -m twine upload dist/* 

# test
python -c "import ichigo.asr as asr; print(asr.__file__)"
```

1. Install python package

```bash
pip install ichigo
```

2. Transcribe with your audio

```python
# Quick one-liner
from ichigo.asr import transcribe
text = transcribe("speech.wav", "transcript.txt")

# Or with more control
from ichigo.asr import IchigoASR
model = IchigoASR(model_name="custom-model", model_path="path/to/model")
text = model.transcribe("speech.wav")
print(text)
```