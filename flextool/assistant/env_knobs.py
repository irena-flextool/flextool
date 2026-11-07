"""The environment variables of the AI assistant (P1 §8.2), registered apart from FlexTool's own.

The assistant has no ``FLEXTOOL_*`` variable (P1-D8): every choice is a flag or a setting.  It reads only the key
variables of the presets, by a name taken from the preset (:func:`flextool.assistant.keys.read_key_env`), so the
name is never a literal next to the read; this registry lists them for the documentation and for
``tests/test_env_knob_registry.py``, which checks that it matches the presets.  Two more variables are named in
the assistant's messages and documentation but read by others: Ollama's context length and keyring's backend
choice.  The variables the provider libraries would read (base addresses, organisation ids, custom headers, log
levels) have no effect and are only described in prose on the page.
"""
from __future__ import annotations

from flextool.run_options import USER_PAGE, EnvKnob


def _key(name: str, service: str) -> EnvKnob:
    return EnvKnob(name, "user", USER_PAGE,
                   f"The key of {service} for the assistant chat. It wins over a key stored in the system keyring "
                   "or the key file.")


ASSISTANT_ENV_KNOBS: dict[str, EnvKnob] = {
    knob.name: knob
    for knob in (
        _key("OPENAI_API_KEY", "OpenAI"),
        _key("ANTHROPIC_API_KEY", "Anthropic"),
        _key("MISTRAL_API_KEY", "Mistral"),
        _key("OPENROUTER_API_KEY", "OpenRouter"),
        _key("GEMINI_API_KEY", "Google Gemini"),
        EnvKnob("OLLAMA_CONTEXT_LENGTH", "user", USER_PAGE,
                "Ollama's own: the context length Ollama loads models with. The assistant needs about 16384; set "
                "it where the Ollama server starts and restart Ollama.", read_by="ollama"),
        EnvKnob("PYTHON_KEYRING_BACKEND", "user", USER_PAGE,
                "keyring's own: which keyring backend stores the keys. FlexTool does not use a backend that "
                "stores keys unencrypted.", read_by="keyring"),
    )
}
