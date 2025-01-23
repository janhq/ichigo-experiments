import os
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load`"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="`do_sample` is set to `False`"
)

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from whisperspeech.vq_stoks import RQBottleneckTransformer

from ichigo.asr.utils import load_model
from ichigo.llm.utils import convert_ids_to_tokens

from .base import VoiceAssistant


class IchigoAssistant(VoiceAssistant):
    def __init__(self, use_speechless=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_speechless:
            self.setup_speechless()
        else:
            self.setup_ichigo()

    def setup_ichigo(self):
        if not os.path.exists("./cache/whisper-vq-stoks-v3-7lang.model"):
            hf_hub_download(
                repo_id="WhisperSpeech/WhisperSpeech",
                filename="whisper-vq-stoks-v3-7lang.model",
                local_dir="./cache/",
            )
        self.vq_model = RQBottleneckTransformer.load_model(
            "./cache/whisper-vq-stoks-v3-7lang.model"
        ).to(self.device)
        self.vq_model.ensure_whisper(self.device)

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

    def setup_speechless(self):
        ichigo_name = "homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth"
        model_size = "merge-medium-vi-2d-2560c-dim64"
        self.ichigo_model = load_model(ichigo_name, model_size)
        self.ichigo_model.ensure_whisper(self.device)
        self.ichigo_model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "homebrewltd/Ichigo-llama3.1-8B-v0.5", cache_dir="./cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "homebrewltd/Ichigo-llama3.1-8B-v0.5",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            cache_dir="./cache",
        )
        self.prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def audio_to_sound_tokens(self, wav, sr, device="cuda"):
        if hasattr(self, "vq_model"):  # Ichigo mode
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.half()
            with torch.no_grad():
                codes = self.vq_model.encode_audio(wav.to(device))
                codes = codes[0].cpu().tolist()
            result = "".join(f"<|sound_{num:04d}|>" for num in codes)
            return f"<|sound_start|>{result}<|sound_end|>"
        else:  # Speechless mode
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            with torch.no_grad():
                codes = self.ichigo_model.quantize(wav.to(device))
                codes = codes[0].cpu().tolist()
            return convert_ids_to_tokens(codes)

    def generate_text(self, audio, max_new_tokens=2048):
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
        else:
            wav = audio["array"].unsqueeze(0)
            sr = audio["sampling_rate"]

        sound_tokens = self.audio_to_sound_tokens(wav, sr)

        if hasattr(self, "pipe"):  # Ichigo mode
            messages = [{"role": "user", "content": sound_tokens}]
            generation_args = {
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            }
            output = self.pipe(messages, **generation_args)
            return output[0]["generated_text"]
        else:  # Speechless mode
            message = self.prompt_template.format(text=sound_tokens)
            input_ids = self.tokenizer([message], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(input_ids.input_ids, generated_ids)
            ]
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
