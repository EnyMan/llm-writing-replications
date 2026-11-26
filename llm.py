"""Shared model client for both replications: OpenRouter, NanoGPT (OpenAI-compatible) and local Ollama.

Keys come from the environment or from `.env` at the repository root (KEY=value lines; real env vars win):
  OPENROUTER_API_KEY   needed for OpenRouter models (plain model ids, e.g. "openai/gpt-5.6-luna")
  NANOGPT_API_KEY      needed for "nanogpt:<id>" models
Ollama models are written "ollama:<name>" and need a local Ollama server on port 11434.
"""
import json, os, re, sys, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
_env = os.path.join(ROOT, ".env")
if os.path.exists(_env):
    for line in open(_env, encoding="utf-8"):
        k, _, v = line.strip().partition("=")
        if k and not k.startswith("#"):
            os.environ.setdefault(k, v.strip().strip('"'))
KEY = os.environ.get("OPENROUTER_API_KEY")
NANO_KEY = os.environ.get("NANOGPT_API_KEY")
# NanoGPT list prices ($ per 1M tokens in, out), for cost accounting where the API reports none
NANO_PRICES = {"z-ai/glm-5.3-flash-uncensored": (0.2, 0.8), "deepseek/deepseek-v4.1-flash-uncensored": (0.33, 1.32),
               "Gemma-4-31B-Gembrain-uncensored-heretic": (0.306, 0.306), "qwen/qwen3.8-27b-antislop": (0.1, 0.6),
               "qwen/qwen3.8-27b": (0.15, 0.7)}


def need(key, name):
    return key or sys.exit(f"Set {name} in .env at the repository root (see .env.example)")


def chat(model, messages, max_tokens, effort):
    """One chat call. effort: 'off' or a reasoning level ('low', 'medium', ...).
    Returns (text, usage dict with cost, reasoning_tokens, finish; provider)."""
    if model.startswith("ollama:"):
        body = {"model": model.removeprefix("ollama:"), "messages": messages, "stream": False,
                "think": effort != "off", "options": {"num_predict": max_tokens}}
        url, headers = "http://localhost:11434/api/chat", {}
    elif model.startswith("nanogpt:"):
        body = {"model": model.removeprefix("nanogpt:"), "messages": messages, "max_tokens": max_tokens,
                "reasoning_effort": "none" if effort == "off" else effort}
        url, headers = "https://nano-gpt.com/api/v1/chat/completions", {"Authorization": f"Bearer {need(NANO_KEY, 'NANOGPT_API_KEY')}"}
    else:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True},
                "reasoning": {"enabled": False} if effort == "off" else {"effort": effort, "exclude": True}}
        url, headers = "https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {need(KEY, 'OPENROUTER_API_KEY')}"}
    req = urllib.request.Request(url, json.dumps(body).encode(), {**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    if model.startswith("ollama:"):
        u = {"prompt_tokens": d.get("prompt_eval_count"), "completion_tokens": d.get("eval_count"), "cost": 0}
        return d["message"].get("content", ""), u, "ollama-local"
    u = d.get("usage", {})
    text = d["choices"][0]["message"].get("content") or ""
    text = re.sub(r"(?s)^.*?</think>\s*", "", text)  # some models leak reasoning before </think>
    u = {**u, "cost": cost_of(model, u), "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
         "finish": d["choices"][0].get("finish_reason")}
    return text, u, d.get("provider")


def cost_of(model, usage):
    """OpenRouter reports usage.cost; for NanoGPT fall back to its list prices."""
    if usage.get("cost") is not None:
        return usage["cost"]
    p_in, p_out = NANO_PRICES.get(model.removeprefix("nanogpt:"), (0, 0))
    return ((usage.get("prompt_tokens") or 0) * p_in + (usage.get("completion_tokens") or 0) * p_out) / 1e6
