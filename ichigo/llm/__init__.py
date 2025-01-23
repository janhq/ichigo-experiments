import torchaudio

from ichigo.llm.ichigo import IchigoAssistant

_default_assistant = None


def get_assistant(**kwargs) -> IchigoAssistant:
    """Get or create default voice assistant instance"""
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = IchigoAssistant(**kwargs)
    return _default_assistant


def process_audio(audio_path, max_new_tokens: int = 2048, **kwargs) -> str:
    """Quick audio processing function using default assistant"""
    assistant = get_assistant(**kwargs)

    wav, sr = torchaudio.load(audio_path)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    audio_dict = {"array": wav, "sampling_rate": sr}

    return assistant.generate_audio(audio_dict, max_new_tokens=max_new_tokens)
