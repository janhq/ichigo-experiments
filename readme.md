# pip install ichigo

## Get Started 

### PyPI

0. Setup package with `python=3.10`

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
import torch, torchaudio
from ichigo.asr.demo.utils import load_model

# Load Ichigo Whisper
ichigo_model = load_model(
        ref="homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth",
        size="merge-medium-vi-2d-2560c-dim64",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
ichigo_model.ensure_whisper(device)
ichigo_model.to(device)

# Inference
wav, sr = torchaudio.load("path/to/your/audio")
if sr != 16000:
   wav = torchaudio.functional.resample(wav, sr, 16000)
transcribe = ichigo_model.inference(wav.to(device))
print(transcribe[0].text)
```