import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Set, Union

import torch
import torchaudio

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
        start_time = time.time()

        # Load and preprocess audio
        wav, sr = torchaudio.load(str(audio_path))
        wav = self.preprocess(wav, sr)

        # Get audio duration in seconds
        duration = wav.shape[1] / 16000  # 16kHz sampling rate

        # Get transcription
        result = self.model.inference(wav)
        transcript = result[0].text

        # Calculate processing speed
        process_time = time.time() - start_time
        rtf = process_time / duration if duration > 0 else 0

        # Save if output path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)

        return transcript, {
            "duration": duration,
            "process_time": process_time,
            "rtf": rtf,
        }

    def transcribe_folder(
        self,
        input_folder: Union[str, Path],
        output_folder: Optional[Union[str, Path]] = None,
        extensions: tuple = (".wav", ".mp3", ".flac"),
    ) -> Dict[str, str]:
        """Transcribe all audio files in a folder.

        Args:
            input_folder: Path to folder containing audio files
            output_folder: Path to save transcripts (defaults to 'transcripts' subfolder)
            extensions: Tuple of valid audio file extensions to process

        Returns:
            Dictionary mapping filenames to their transcripts
        """
        input_folder = Path(input_folder)
        if output_folder is None:
            output_folder = input_folder / "transcripts"
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)

        # Track files for reporting
        audio_files: Set[Path] = set()
        non_audio_files: Set[Path] = set()

        # Categorize files
        for file_path in input_folder.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() in extensions:
                    audio_files.add(file_path)
                else:
                    non_audio_files.add(file_path)

        # Warn about non-audio files
        if non_audio_files:
            warnings.warn(
                f"Found {len(non_audio_files)} non-audio files that will be skipped:\n"
                f"{', '.join(f.name for f in non_audio_files)}"
            )

        # Warn if no audio files found
        if not audio_files:
            warnings.warn(
                f"No audio files found with extensions {extensions} in {input_folder}"
            )
            return {}

        # Process audio files
        results = {}
        for audio_file in sorted(audio_files):  # Sort for consistent processing order
            output_path = output_folder / f"{audio_file.stem}.txt"

            try:
                transcript = self.transcribe(audio_file, str(output_path))
                results[audio_file.name] = transcript
                print(f"Successfully transcribed: {audio_file.name}")

            except Exception as e:
                error_msg = f"Error processing {audio_file.name}: {str(e)}"
                print(f"ERROR: {error_msg}")
                results[audio_file.name] = f"ERROR: {error_msg}"

        # Print summary
        print(f"\nTranscription Summary:")
        print(f"- Total files in directory: {len(non_audio_files) + len(audio_files)}")
        print(f"- Audio files processed: {len(audio_files)}")
        print(f"- Non-audio files skipped: {len(non_audio_files)}")
        print(
            f"- Successful transcriptions: {sum(1 for v in results.values() if not v.startswith('ERROR'))}"
        )
        print(
            f"- Failed transcriptions: {sum(1 for v in results.values() if v.startswith('ERROR'))}"
        )

        return results
