from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel

from ichigo.asr import get_model, release_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model to GPU at startup
    get_model()
    release_model()
    yield


app = FastAPI(
    title="Ichigo ASR API",
    description="API for audio transcription using Ichigo ASR",
    version="0.0.1",
    lifespan=lifespan,
)


class TranscriptionsModelName(str, Enum):
    ichigo = "ichigo"


@app.post("/v1/audio/transcriptions")
def _(
    file: Annotated[UploadFile, File()],
    model: Annotated[TranscriptionsModelName, Form()],
):
    """
    Transcribe an audio file uploaded via HTTP POST request

    Args:
        file: Audio file to transcribe
    """
    wav, sr = torchaudio.load(file.file)
    if wav.shape[0] > 1:  # convert multi-channel audio to mono
        wav = wav.mean(0, keepdim=True)

    model_ = get_model()
    wav = model_.preprocess(wav, sr)

    token_ids = model_.model.quantize(wav)
    embeds = model_.model.dequantize(token_ids)
    output = model_.model.whmodel[0].decode(embeds, model_.model.decoding_options)[0].text
    release_model()
    return dict(text=output)


@app.post("/s2r")
def _(file: UploadFile = File(...)):
    wav, sr = torchaudio.load(file.file)
    if wav.shape[0] > 1:  # convert multi-channel audio to mono
        wav = wav.mean(0, keepdim=True)

    model = get_model()
    wav = model.preprocess(wav, sr)
    token_ids = model.model.quantize(wav).squeeze(0).tolist()
    release_model()

    output = ''.join(f"<|sound_{tok:04d}|>" for tok in token_ids)
    output = f"<|sound_start|>{output}<|sound_end|>"

    return dict(model_name=model.model_name, tokens=output)


class R2TRequest(BaseModel):
    tokens: str


@app.post("/r2t")
def _(req: R2TRequest):
    """tokens will have format <|sound_start|><|sound_0000|><|sound_end|>
    """
    token_ids = [int(x) for x in req.tokens.split("|><|sound_")[1:-1]]
    token_ids = torch.tensor(token_ids).unsqueeze(0)

    model = get_model().model
    token_ids = token_ids.to(model.device)
    embeds = model.dequantize(token_ids)
    output = model.whmodel[0].decode(embeds, model.decoding_options)[0].text
    release_model()

    return dict(text=output)
