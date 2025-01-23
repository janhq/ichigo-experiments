import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load`"
)


import torch
import torchaudio

from .utils import load_model

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union
import torch
import yaml


class BaseASRModel(ABC):
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sampling_rate = config["sampling_rate"]
        self.s2r_model = self.load_model(config['s2r_model_path'], config['s2r_model_name'])
        self.quantizer = self.load_model(config['quantizer_model_path'], config['quantizer_model_name'])
        self.r2t_model = self.load_model(config['r2t_model_path'], config['r2t_model_name'])
        self.set_device()
        self.set_inference_mode()
    
    @abstractmethod
    def set_device(self) -> str:
        """Set the device for the model."""
        self.s2r_model.to(self.device)
        self.quantizer.to(self.device)
        self.r2t_model.to(self.device)
        return
    
    @abstractmethod
    def set_inference_mode(self):
        """Set all models to inference mode."""
        self.s2r_model.eval()
        self.quantizer.eval()
        self.r2t_model.eval()
        return

    @abstractmethod
    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """Preprocess audio to match model requirements. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def transcribe(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        extensions: tuple = (".wav", ".mp3", ".flac"),
    ) -> Union[str, Dict[str, str]]:
        """Transcribe audio file or folder of audio files.

        Args:
            input_path: Path to audio file or folder containing audio files
            output_path: Path to save transcript(s). If input is folder, creates 'transcripts' subfolder
            extensions: Tuple of valid audio file extensions to process (only used for folder input)

        Returns:
            For single file: transcript string and metadata dict
            For folder: dictionary mapping filenames to their transcripts
        """
        input_path = Path(input_path)

        # Handle single file by treating it as a folder with one file
        if input_path.is_file():
            if not input_path.suffix.lower() in extensions:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")
            audio_files = [input_path]
        elif input_path.is_dir():
            audio_files = [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ]
            if not audio_files:
                warnings.warn(
                    f"No audio files found with extensions {extensions} in {input_path}"
                )
                return {}
            
            non_audio = [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() not in extensions
            ]
            if non_audio:
                warnings.warn(
                    f"Found {len(non_audio)} non-audio files that will be skipped:\n"
                    f"{', '.join(f.name for f in non_audio)}"
                )
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
        
        #Construct output_path
        if output_path is None:
            output_path = input_path.parent / "transcriptions"
        elif output_path.is_dir():
            output_path = Path(output_path)
        else:
            raise ValueError(f"Output path {output_path} is a file, expected a directory.")

        results = self._transcribe_files(audio_files, output_path)

        return results if len(audio_files) > 1 else (results[audio_files[0].name], results[audio_files[0].name + "_metadata"])
   
    @abstractmethod
    def _inference(self, wav):
        """Transcribe a list of audio files and save results to a file.

        Args:
            wav: wav file 

        Returns:
            transcript: a string representing the transcription of the wav

        This method should be overwritten if there are any special cases/processing needed for the specific model
        """
        transcript = self.r2t(self.quantizer(self.s2r(wav)))
        return transcript

    @abstractmethod
    def _transcribe_files(self, audio_files: list, output_path: Optional[Path]) -> Dict[str, str]:
        """Transcribe a list of audio files and save results to a file.

        Args:
            audio_files: List of audio file paths to transcribe.
            output_path: Path to a directory to save the transcription results.

        Returns:
            Dictionary mapping filenames to their transcripts.
        """
        results = {}

        # Create or open the transcription file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for audio_file in sorted(audio_files):
                try:
                    start_time = time.time()
                    wav, sr = torchaudio.load(str(audio_file))
                    wav = self.preprocess(wav, sr)
                    duration = wav.shape[1] / 16000

                    transcript = self._inference(wav, return_stoks=self.return_stoks)

                    process_time = time.time() - start_time
                    metadata = {
                        "duration": duration,
                        "process_time": process_time,
                        "rtf": process_time / duration if duration > 0 else 0,
                    }

                    results[audio_file.name] = transcript
                    results[audio_file.name + "_metadata"] = metadata
                    f.write(f"{audio_file.name}\t{transcript}\n")
                    print(f"Successfully transcribed: {audio_file.name}")

                except Exception as e:
                    error_msg = f"Error processing {audio_file.name}: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    results[audio_file.name] = f"ERROR: {error_msg}"
                    f.write(f"{audio_file.name}\tERROR: {error_msg}\n")

        return results
    
class IchigoASR:
    def __init__(
        self,
        model_name: str = "merge-medium-vi-2d-2560c-dim64",
        model_path: str = "homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth",
        return_stoks: bool = False,
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.return_stoks = return_stoks

        self.model = load_model(
            ref=model_path, size=model_name, return_stoks=return_stoks
        )
        self.model.ensure_whisper(self.device)
        self.model.to(self.device)

    def preprocess(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess audio to match model requirements"""
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        return audio.to(self.device)

    def transcribe(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = "transcription.txt",
        extensions: tuple = (".wav", ".mp3", ".flac"),
    ) -> Union[str, Dict[str, str]]:
        """Transcribe audio file or folder of audio files.

        Args:
            input_path: Path to audio file or folder containing audio files
            output_path: Path to save transcript(s). If input is folder, creates 'transcripts' subfolder
            extensions: Tuple of valid audio file extensions to process (only used for folder input)

        Returns:
            For single file: transcript string and metadata dict
            For folder: dictionary mapping filenames to their transcripts
        """
        input_path = Path(input_path)

        # Handle single file
        if input_path.is_file():
            if not input_path.suffix.lower() in extensions:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")

            start_time = time.time()
            wav, sr = torchaudio.load(str(input_path))
            wav = self.preprocess(wav, sr)
            duration = wav.shape[1] / 16000

            result = self.model.inference(wav, return_stoks=self.return_stoks)

            if self.return_stoks:
                return result

            transcript = result[0].text

            process_time = time.time() - start_time
            metadata = {
                "duration": duration,
                "process_time": process_time,
                "rtf": process_time / duration if duration > 0 else 0,
            }

            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcript)

            return transcript, metadata

        # Handle folder
        elif input_path.is_dir():
            if output_path is None:
                output_path = input_path / "transcription.txt"
            else:
                output_path = Path(output_path)
                if output_path.is_dir():
                    output_path = output_path / "transcription.txt"

            audio_files = [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ]
            non_audio = [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() not in extensions
            ]

            if non_audio:
                warnings.warn(
                    f"Found {len(non_audio)} non-audio files that will be skipped:\n"
                    f"{', '.join(f.name for f in non_audio)}"
                )

            if not audio_files:
                warnings.warn(
                    f"No audio files found with extensions {extensions} in {input_path}"
                )
                return {}

            results = {}

            # Create or open the transcription file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for audio_file in sorted(audio_files):
                    try:
                        transcript, _ = self.transcribe(audio_file, None)
                        results[audio_file.name] = transcript
                        f.write(f"{audio_file.name}\t{transcript}\n")
                        print(f"Successfully transcribed: {audio_file.name}")
                    except Exception as e:
                        error_msg = f"Error processing {audio_file.name}: {str(e)}"
                        print(f"ERROR: {error_msg}")
                        results[audio_file.name] = f"ERROR: {error_msg}"
                        f.write(f"{audio_file.name}\tERROR: {error_msg}\n")

            success = sum(1 for v in results.values() if not v.startswith("ERROR"))
            failed = sum(1 for v in results.values() if v.startswith("ERROR"))
            print(f"\nTranscription Summary:")
            print(f"- Total files in directory: {len(audio_files) + len(non_audio)}")
            print(f"- Audio files processed: {len(audio_files)}")
            print(f"- Non-audio files skipped: {len(non_audio)}")
            print(f"- Successful transcriptions: {success}")
            print(f"- Failed transcriptions: {failed}")

            return results

        else:
            raise ValueError(f"Input path does not exist: {input_path}")
