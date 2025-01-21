from ichigo.asr.transcriber import IchigoASR

_default_model = None


def get_model(**kwargs) -> IchigoASR:
    """Get or create default ASR model instance"""
    global _default_model
    if _default_model is None:
        _default_model = IchigoASR(**kwargs)
    return _default_model


def transcribe(audio_path: str, output_path: str = None, **kwargs) -> str:
    """Quick transcription function using default model"""
    model = get_model(**kwargs)
    return model.transcribe(audio_path, output_path)


def transcribe_folder(
    input_folder: str,
    output_folder: str = None,
    extensions: tuple = (".wav", ".mp3", ".flac"),
    **kwargs,
) -> dict[str, str]:
    """Quick folder transcription function using default model"""
    model = get_model(**kwargs)
    return model.transcribe_folder(input_folder, output_folder, extensions)
