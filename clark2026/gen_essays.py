"""Essay setting of Clark, Martin & Mahowald (2026), arXiv 2610.29162, appendix "Essay".

Each model writes N essays for each of the seven PERSUADE 2.0 independent assignments.
System prompt: "You are {assistant}, an AI assistant made by {vendor}."  User message: the assignment text.
Reasoning at "medium" (the paper's Pi "medium" level), sampling at provider defaults.

usage (from the repo root): uv run python clark2026/gen_essays.py [N_PER_ASSIGNMENT] model [model ...]
Output: clark2026/out/essays/<model>.jsonl, resumable. Hard stop at BUDGET over clark2026/out/**.
"""
import glob, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import chat  # noqa: E402  (loads .env from the repo root; OpenRouter/NanoGPT/Ollama routing)

BUDGET = 10.0  # hard stop on total spend over out/**/*.jsonl (raised from $5 during the original run)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "essays")
# (assistant, vendor) for the system prompt; finetunes keep their base vendor's name unless they have their own
WHO = {"openai/gpt-5.6-luna": ("ChatGPT", "OpenAI"), "z-ai/glm-5": ("GLM", "Z.ai"),
       "moonshotai/kimi-k2.5": ("Kimi", "Moonshot AI"), "meta-llama/llama-3.1-8b-instruct": ("Llama", "Meta"),
       # niche models (our extension): finetunes keep their base family's assistant name unless they have their own
       "nanogpt:z-ai/glm-5.3-flash-uncensored": ("GLM", "Z.ai"),
       "nanogpt:deepseek/deepseek-v4.1-flash-uncensored": ("DeepSeek", "DeepSeek"),
       "nanogpt:Gemma-4-31B-Gembrain-uncensored-heretic": ("Gemma", "Google"),
       "nanogpt:qwen/qwen3.8-27b-antislop": ("Qwen", "Alibaba Cloud"), "nanogpt:qwen/qwen3.8-27b": ("Qwen", "Alibaba Cloud"),
       "ollama:richardyoung/hemmingway-1:Q4_K_M": ("Hemmingway", "Altworld"),
       "mistralai/mistral-small-3.1-24b-instruct": ("Mistral", "Mistral AI")}
# the paper's frontier models ran at "medium"; its local models (Llama, Olmo) had no reasoning. The niche models are
# small/open like the latter, so reasoning off where the provider allows it, lowest level where it is mandatory.
EFFORT = {"meta-llama/llama-3.1-8b-instruct": "off", "nanogpt:z-ai/glm-5.3-flash-uncensored": "low",
          "nanogpt:deepseek/deepseek-v4.1-flash-uncensored": "off", "nanogpt:Gemma-4-31B-Gembrain-uncensored-heretic": "off",
          "nanogpt:qwen/qwen3.8-27b-antislop": "off", "nanogpt:qwen/qwen3.8-27b": "off",
          "ollama:richardyoung/hemmingway-1:Q4_K_M": "off", "mistralai/mistral-small-3.1-24b-instruct": "off"}
lock = threading.Lock()


def spent():
    """Spend so far on essays and chat, which share BUDGET (coding has its own, gen_code.CODE_BUDGET)."""
    return sum(json.loads(l).get("cost") or 0 for d in ("essays", "chat")
               for f in glob.glob(os.path.join(HERE, "out", d, "**", "*.jsonl"), recursive=True)
               for l in open(f, encoding="utf-8"))


def one(model, a, rep, path, state):
    if state["spent"] >= BUDGET:
        return
    assistant, vendor = WHO[model]
    msgs = [{"role": "system", "content": f"You are {assistant}, an AI assistant made by {vendor}."},
            {"role": "user", "content": a["assignment"]}]
    for attempt in range(3):  # the paper: up to three attempts per call
        try:
            text, u, provider = chat(model, msgs, 32000, EFFORT.get(model, "medium"))
            break
        except Exception as e:
            if attempt == 2:
                print(f"  {model} {a['prompt_name']} #{rep}: {e}")
                return
    rec = {"prompt_name": a["prompt_name"], "rep": rep, "model": model, "text": text, "provider": provider, **u}
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        state["spent"] += rec.get("cost") or 0


def main():
    n = int(sys.argv[1])
    assignments = json.load(open(os.path.join(HERE, "data", "assignments.json"), encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    state = {"spent": spent()}
    print(f"already spent ${state['spent']:.2f} of ${BUDGET}")
    for model in sys.argv[2:]:
        path = os.path.join(OUT, model.replace("/", "__").replace(":", "_") + ".jsonl")
        done = {(r["prompt_name"], r["rep"]) for r in map(json.loads, open(path, encoding="utf-8")) if r["text"].strip()} \
            if os.path.exists(path) else set()
        todo = [(a, rep) for a in assignments for rep in range(n) if (a["prompt_name"], rep) not in done]
        with ThreadPoolExecutor(4 if model.startswith("ollama:") else 8) as ex:
            for a, rep in todo:
                ex.submit(one, model, a, rep, path, state)
        print(f"{model}: +{len(todo)} essays, spent ${state['spent']:.2f}")
        if state["spent"] >= BUDGET:
            print("BUDGET reached")
            break


if __name__ == "__main__":
    main()
