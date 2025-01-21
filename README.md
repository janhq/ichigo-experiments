# Ichigo Package

## PyPI

<!-- 
python -m build
pip install dist/ichigo-0.0.1-py3-none-any.whl
python -c "import ichigo.asr as asr; print(asr.__file__)" 
python -c "from ichigo.asr import transcribe; text = transcribe('speech.wav', 'transcript.txt'); print(text)"
python -c "from ichigo.asr import transcribe; text = transcribe('/root/ichigo-experiments/test', 'transcript.txt'); print(text)"
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

2. Transcribe with your audio file

```python
from ichigo.asr import transcribe
text, metadata = transcribe("path/to/your/file/or/folder")
```