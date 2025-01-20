from pathlib import Path
import torch
import torchaudio
from typing import Optional, Union

from .utils import load_model


class IchigoASR:
    def __init__(
        self,
        model_name: str = "merge-medium-vi-2d-2560c-dim64",
        model_path: str = "homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load and initialize model
        self.model = load_model(ref=model_path, size=model_name)
        self.model.ensure_whisper(self.device)
        self.model.to(self.device)

    def preprocess(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess audio to match model requirements"""
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        return audio.to(self.device)

    def transcribe(
        self, audio_path: Union[str, Path], output_path: Optional[str] = None
    ) -> str:
        """Transcribe audio file and optionally save to text file"""
        # Load and preprocess audio
        wav, sr = torchaudio.load(str(audio_path))
        wav = self.preprocess(wav, sr)

        # Get transcription
        result = self.model.inference(wav)
        transcript = result[0].text

        # Save if output path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)

        return transcript
