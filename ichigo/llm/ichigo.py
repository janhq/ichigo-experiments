import os
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load`"
)

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from whisperspeech.vq_stoks import RQBottleneckTransformer

from .base import VoiceAssistant


class IchigoAssistant(VoiceAssistant):
    def __init__(self):
        device = "cuda"

        if not os.path.exists("./cache/whisper-vq-stoks-v3-7lang.model"):
            hf_hub_download(
                repo_id="WhisperSpeech/WhisperSpeech",
                filename="whisper-vq-stoks-v3-7lang.model",
                local_dir="./cache/",
            )
        self.vq_model = RQBottleneckTransformer.load_model(
            "./cache/whisper-vq-stoks-v3-7lang.model"
        ).to(device)
        self.vq_model.ensure_whisper(device)

        tokenizer = AutoTokenizer.from_pretrained(
            "homebrewltd/Ichigo-llama3.1-s-instruct-v0.4", cache_dir="./cache"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "homebrewltd/Ichigo-llama3.1-s-instruct-v0.4",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            cache_dir="./cache",
        )

        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def audio_to_sound_tokens(self, wav, sr, device="cuda"):
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        wav = wav.half()

        with torch.no_grad():
            codes = self.vq_model.encode_audio(wav.to(device))
            codes = codes[0].cpu().tolist()

        result = "".join(f"<|sound_{num:04d}|>" for num in codes)
        return f"<|sound_start|>{result}<|sound_end|>"

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
        else:
            wav = audio["array"].unsqueeze(0)
            sr = audio["sampling_rate"]

        sound_tokens = self.audio_to_sound_tokens(wav, sr)

        messages = [
            {"role": "user", "content": sound_tokens},
        ]

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
        }

        output = self.pipe(messages, **generation_args)
        return output[0]["generated_text"]
