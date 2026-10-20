import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pal.pipeline import Pipeline
from pal.llm_wrapper.components.base import WrapperConfig


def run_naive(pipeline: Pipeline, system_prompt: str, message: str) -> dict:
    core = pipeline.llm_wrapper.core
    tokens = core.token_count(system_prompt) + core.token_count(message)
    response = pipeline.llm_wrapper.syscall_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ])
    return {"tokens": tokens, "latency": response.meta.get("latency", 0)}


def run_optimized(pipeline: Pipeline, system_prompt: str, message: str) -> dict:
    pipeline.run(system_prompt, message)
    last = pipeline.metrics.records[-1]
    return {"tokens": last.tokens_optimized, "latency": last.latency}


def main(config_path: str = None, scenario_path: str = None):
    config = WrapperConfig()
    scenarios = json.loads(Path(scenario_path).read_text()) if scenario_path else [
        {"system": "You are a helpful assistant.", "message": "Explain caching in one line."}
    ]

    pipeline = Pipeline(config)
    results = []
    for s in scenarios:
        naive = run_naive(pipeline, s["system"], s["message"])
        optimized = run_optimized(pipeline, s["system"], s["message"])
        results.append({"scenario": s["message"], "naive": naive, "optimized": optimized})

    print(json.dumps({"results": results, "summary": pipeline.summary()}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
