"""The Local LLM component: models served on this machine (Ollama, vLLM, LM Studio, llama.cpp) and the in-process Hugging Face model."""
from pal.local_llm.components.huggingface_local import HuggingFaceLocalLLM
from pal.local_llm.components.local_llm import DEFAULT_URLS, LOCAL_BACKENDS, LocalLLM

__all__ = ["DEFAULT_URLS", "LOCAL_BACKENDS", "HuggingFaceLocalLLM", "LocalLLM"]
