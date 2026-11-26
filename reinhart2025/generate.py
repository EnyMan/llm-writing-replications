"""Generate HAP-E style continuations (OpenRouter, NanoGPT or local Ollama; client in ../llm.py).

usage (from reinhart2025/): uv run python generate.py [--nocount] [N_PER_GENRE] [model ...]
  --nocount: use PROMPT + NOCOUNT for every model (calibration run)
  N_PER_GENRE: docs per genre (default 150 = all 900). Use 2 for a pilot (12 docs/model).
  models: default = models_selected.txt
Output: out/<model-slug>[@nocount].jsonl, one line per doc. Resumable: done doc_ids are skipped.
Hard stop when total spend across out/**/*.jsonl reaches BUDGET.
"""
import json, os, sys, glob, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import KEY, NANO_KEY, need, cost_of  # noqa: E402

BUDGET = 20.0
PROMPT = ("In the same style, tone, and diction of the following text, complete the next 500 words, "
          "generate exactly 500 words, and note that the text does not necessarily end after the generated words:")
# Reasoning models recount drafts word by word until they run out of tokens. Added only for
# NOCOUNT_MODELS; calibration (--nocount on non-reasoning models) checks it doesn't shift style.
NOCOUNT = " Do not count the words; an approximate length of about 500 words is fine."
NOCOUNT_MODELS = {"minimax/minimax-m3.1-flash", "baidu/ernie-5.5-flash-preview:free"}
# NanoGPT models are written "nanogpt:<id>". Lowest reasoning effort each one offers (none = off);
# models not listed get no reasoning parameter. (List prices for cost accounting live in llm.py.)
NANO_EFFORT = {"z-ai/glm-5.3-flash-uncensored": "low", "deepseek/deepseek-v4.1-flash-uncensored": "none",
               "Gemma-4-31B-Gembrain-uncensored-heretic": "none"}
lock = threading.Lock()
# Per-model reasoning (default: off). minimax/ernie ignore "off" -> minimal effort + room.
# GLM 5.3 Flash: a 1024-token reasoning cap keeps it from endless recounting. Luna reasons by default (a deliberate choice: it is how the model ships).
REASONING = {
    "minimax/minimax-m3.1-flash": {"effort": "minimal", "exclude": True},
    "baidu/ernie-5.5-flash-preview:free": {"effort": "minimal", "exclude": True},
    "z-ai/glm-5.3-flash": {"max_tokens": 1024, "exclude": True},
    "openai/gpt-6-luna": {"enabled": True, "exclude": True},
    "openai/gpt-5.6-luna": {"enabled": True, "exclude": True},
}


def spent():
    total = 0.0
    for f in glob.glob("out/**/*.jsonl", recursive=True):  # includes out/_discarded
        for line in open(f, encoding="utf-8"):
            total += json.loads(line).get("cost") or 0
    return total


def call(model, text, reasoning, nocount):
    content = f"{PROMPT}{NOCOUNT if nocount else ''}\n\n{text}"
    if model.startswith("ollama:"):  # local; native API so thinking can be switched off; model's own sampling defaults
        body = {"model": model.removeprefix("ollama:"), "messages": [{"role": "user", "content": content}],
                "stream": False, "think": False, "options": {"num_predict": 1500}}
        req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            o = json.load(r)
        return {"choices": [{"message": {"content": o["message"].get("content", "")}, "finish_reason": o.get("done_reason")}],
                "usage": {"prompt_tokens": o.get("prompt_eval_count"), "completion_tokens": o.get("eval_count"), "cost": 0},
                "provider": "ollama-local"}
    if model.startswith("nanogpt:"):  # OpenAI-compatible; reasoning via reasoning_effort
        mid = model.removeprefix("nanogpt:")
        body = {"model": mid, "messages": [{"role": "user", "content": content}], "max_tokens": 1500}
        if mid in NANO_EFFORT:
            body["reasoning_effort"] = NANO_EFFORT[mid]
            body["max_tokens"] = 8000 if NANO_EFFORT[mid] != "none" else 1500
        url, key = "https://nano-gpt.com/api/v1/chat/completions", need(NANO_KEY, "NANOGPT_API_KEY")
    else:
        if model in REASONING:
            reasoning, max_tokens = REASONING[model], 8000
        else:
            max_tokens = 1500
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens, "reasoning": reasoning, "usage": {"include": True}}
        url, key = "https://openrouter.ai/api/v1/chat/completions", need(KEY, "OPENROUTER_API_KEY")
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def one(model, doc_id, text, path, state, nocount):
    if state["stop"] or state.get(f"daily:{model}"):
        return
    for attempt in range(5):
        try:
            # reasoning off; fall back to minimal effort for models where reasoning is mandatory
            try:
                d = call(model, text, {"enabled": False}, nocount)
            except urllib.error.HTTPError as e:
                if e.code != 400:
                    raise
                d = call(model, text, {"effort": "minimal", "exclude": True}, nocount)
            break
        except urllib.error.HTTPError as e:
            body = e.read()
            if e.code == 429 and b"per-day" in body:  # free-tier daily quota: retrying won't help
                if not state.get(f"daily:{model}"):
                    print(f"  {model}: free daily limit reached, skipping until reset: {body[:120]!r}")
                state[f"daily:{model}"] = True
                return
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(10 * 2 ** attempt)  # free models are rate limited
                continue
            print(f"  {model} {doc_id}: HTTP {e.code} {body[:200]!r}")
            return
        except Exception as e:  # timeouts, connection resets
            if attempt < 4:
                time.sleep(5)
                continue
            print(f"  {model} {doc_id}: {e}")
            return
    u = d.get("usage", {})
    rec = {"doc_id": doc_id, "model": model, "text": d["choices"][0]["message"].get("content") or "",
           "finish": d["choices"][0].get("finish_reason"), "cost": cost_of(model, u),
           "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
           "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "provider": d.get("provider"), "prompt": "nocount" if nocount else "orig"}
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        state["spent"] += rec["cost"] or 0
        if state["spent"] >= BUDGET:
            state["stop"] = True
            print(f"BUDGET ${BUDGET} reached, stopping")


def main():
    args = sys.argv[1:]
    force_nocount = "--nocount" in args
    args = [a for a in args if a != "--nocount"]
    n = int(args[0]) if args else 150
    models = args[1:] or open("models_selected.txt").read().split()
    df = pd.read_parquet("data/mini_parallel.parquet")
    docs = df[df.source == "human-chunk-1"].sort_values("doc_id")
    docs = docs.groupby(docs.doc_id.str.split("_").str[0]).head(n)
    os.makedirs("out", exist_ok=True)
    state = {"spent": spent(), "stop": False}
    print(f"already spent ${state['spent']:.2f}; {len(docs)} docs x {len(models)} models")
    for model in models:
        nocount = force_nocount or model in NOCOUNT_MODELS
        path = f"out/{model.replace('/', '__').replace(':', '_')}{'@nocount' if nocount else ''}.jsonl"
        # empty outputs (model thought until the token limit) stay for cost accounting but get retried
        done = ({r["doc_id"] for r in map(json.loads, open(path, encoding="utf-8")) if r["text"].strip()}
                if os.path.exists(path) else set())
        todo = docs[~docs.doc_id.isin(done)]
        workers = 2 if model.endswith(":free") else 4 if model.startswith("ollama:") else 16
        with ThreadPoolExecutor(workers) as ex:
            for r in todo.itertuples():
                ex.submit(one, model, r.doc_id, r.text, path, state, nocount)
        print(f"{model}: +{len(todo)} docs, total spent ${state['spent']:.2f}")
        if state["stop"]:
            break


if __name__ == "__main__":
    main()
