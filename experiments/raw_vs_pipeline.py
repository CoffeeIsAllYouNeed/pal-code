"""
Fair comparison: RAW = naive chatbot that resends the entire conversation
history every turn, unbounded. PIPELINE = pal Pipeline, which caps
resent history via paging (hot_window) instead of sending everything.

Ollama has no server-side cache across separate HTTP requests, so the
only real, honest savings mechanism available here is bounding what we
resend as the conversation grows -- not pretending we can skip content
the model actually needs to see.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pal.llm_wrapper.components.base import WrapperConfig
from pal.llm_wrapper.pipeline import LLMWrapperPipeline
from pal.pipeline import Pipeline

SYSTEM_PROMPT = "You are a helpful assistant. Keep answers to one short sentence."

TURNS = [
    "Tell me a fact about the moon.",
    "Tell me a fact about Mars.",
    "Tell me a fact about Jupiter.",
    "Tell me a fact about Saturn.",
    "Tell me a fact about Venus.",
    "Tell me a fact about Mercury.",
    "Tell me a fact about Neptune.",
    "Tell me a fact about Uranus.",
    "Now summarize what we've covered so far.",
]


def run_raw(config: WrapperConfig) -> list[int]:
    llm_wrapper = LLMWrapperPipeline(config)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    prompt_tokens_per_turn = []

    for turn in TURNS:
        history.append({"role": "user", "content": turn})
        resp = llm_wrapper.syscall_chat(history)
        prompt_tokens_per_turn.append(resp.prompt_tokens)
        history.append({"role": "assistant", "content": resp.text})

    return prompt_tokens_per_turn


def run_pipeline(config: WrapperConfig) -> list[int]:
    pipeline = Pipeline(config)
    prompt_tokens_per_turn = []

    for turn in TURNS:
        pipeline.run(SYSTEM_PROMPT, turn)
        prompt_tokens_per_turn.append(pipeline.metrics.records[-1].tokens_optimized)

    return prompt_tokens_per_turn


def main():
    config = WrapperConfig(backend="ollama", model="llama3")

    print("Running RAW (unbounded history)...")
    raw = run_raw(config)

    print("Running PIPELINE (paged history)...")
    piped = run_pipeline(config)

    print("\nturn | raw_prompt_tokens | pipeline_prompt_tokens | saved")
    total_saved = 0
    for i, (r, p) in enumerate(zip(raw, piped), 1):
        saved = r - p
        total_saved += saved
        print(f"{i:>4} | {r:>18} | {p:>23} | {saved:>5}")

    print(f"\nTotal tokens saved across conversation: {total_saved}")
    print(f"Raw total: {sum(raw)} | Pipeline total: {sum(piped)}")


if __name__ == "__main__":
    main()
