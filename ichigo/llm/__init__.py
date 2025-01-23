from ichigo.llm.ichigo import IchigoAssistant

_default_assistant = None


def get_assistant(**kwargs) -> IchigoAssistant:
    """Get or create default voice assistant instance"""
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = IchigoAssistant(**kwargs)
    return _default_assistant


def process_audio(audio_input, max_new_tokens: int = 2048, **kwargs) -> str:
    """Quick audio processing function using default assistant"""
    assistant = get_assistant(**kwargs)
    return assistant.generate_text(audio_input, max_new_tokens=max_new_tokens)
