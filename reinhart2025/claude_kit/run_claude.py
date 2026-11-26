"""Generate HAP-E continuations with Claude models through `claude -p` (subscription auth).

Python 3.9+ stdlib only. Needs the `claude` CLI on PATH, logged in to the account you want to use.

usage:
  python run_claude.py check                         # 1 tiny call per condition; prints CLI version + input tokens
  python run_claude.py run [--models a,b] [--conditions minimal,default] [--n-per-genre 150] [--workers 4] [--effort low]

Conditions:
  minimal  own one-line system prompt, no tools, no settings/hooks/plugins/MCP, empty working dir
  default  Claude Code's built-in system prompt + built-in tools, but no settings/hooks/plugins/MCP/CLAUDE.md
           from this machine (so results do not depend on the machine), empty working dir
Output: out/<model>@<condition>.jsonl (one line per doc). Resumable: re-run the same command.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = ("In the same style, tone, and diction of the following text, complete the next 500 words, "
          "generate exactly 500 words, and note that the text does not necessarily end after the generated words:")
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2"]
MINIMAL_SYSTEM = "You are a helpful assistant."
lock = threading.Lock()

CLAUDE = shutil.which("claude") or sys.exit("`claude` CLI not found on PATH")
EMPTY_DIR = tempfile.mkdtemp(prefix="claude_kit_cwd_")  # no CLAUDE.md / project files to discover


def cmd(model, condition, effort):
    c = [CLAUDE, "-p", "--model", model, "--output-format", "json", "--no-session-persistence",
         "--setting-sources", "", "--strict-mcp-config"]
    if effort:
        c += ["--effort", effort]
    if condition == "minimal":
        c += ["--system-prompt", MINIMAL_SYSTEM, "--tools", ""]
    elif condition != "default":
        sys.exit(f"unknown condition {condition}")
    return c


def call(model, condition, effort, prompt, timeout=900):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # force subscription auth
    p = subprocess.run(cmd(model, condition, effort), input=prompt, capture_output=True, text=True,
                       encoding="utf-8", cwd=EMPTY_DIR, timeout=timeout, env=env)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"exit {p.returncode}: {(p.stdout or p.stderr)[:300]}")


def record(doc_id, model, condition, effort, d):
    u = d.get("usage") or {}
    return {"doc_id": doc_id, "model": model, "condition": condition, "effort": effort,
            "text": d.get("result") or "", "is_error": d.get("is_error"), "num_turns": d.get("num_turns"),
            "input_tokens": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                            + (u.get("cache_read_input_tokens") or 0),
            "output_tokens": u.get("output_tokens"),
            "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
            "cost": 0, "list_cost_usd": d.get("total_cost_usd"), "prompt": "orig"}


def one(doc, model, condition, effort, path):
    for attempt in range(4):
        try:
            d = call(model, condition, effort, f"{PROMPT}\n\n{doc['text']}")
            if d.get("is_error") and attempt < 3:  # rate limit / overload: back off and retry
                print(f"  {model}@{condition} {doc['doc_id']}: error {str(d.get('result'))[:150]!r}, retrying")
                time.sleep(60 * (attempt + 1))
                continue
            rec = record(doc["doc_id"], model, condition, effort, d)
            with lock, open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            return
        except Exception as e:
            if attempt < 3:
                time.sleep(30)
                continue
            print(f"  {model}@{condition} {doc['doc_id']}: giving up: {e}")


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["doc_id"] for r in map(json.loads, f) if r["text"].strip() and not r.get("is_error")}


def run(a):
    with open(os.path.join(HERE, "prompts.jsonl"), encoding="utf-8") as f:
        docs = [json.loads(l) for l in f]
    per = {}
    docs = [d for d in docs if per.setdefault(d["doc_id"].split("_")[0], []).append(d) or
            len(per[d["doc_id"].split("_")[0]]) <= a.n_per_genre]
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    version = subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(HERE, "out", "run_info.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "claude_version": version,
                            "args": vars(a)}) + "\n")
    for model in a.models.split(","):
        for condition in a.conditions.split(","):
            path = os.path.join(HERE, "out", f"{model}@{condition}.jsonl")
            todo = [d for d in docs if d["doc_id"] not in done_ids(path)]
            print(f"{model}@{condition}: {len(todo)} to do", flush=True)
            with ThreadPoolExecutor(a.workers) as ex:
                for d in todo:
                    ex.submit(one, d, model, condition, a.effort, path)
            print(f"{model}@{condition}: {len(done_ids(path))}/{len(docs)} done", flush=True)


def check(a):
    print(subprocess.run([CLAUDE, "--version"], capture_output=True, text=True).stdout.strip())
    for condition in ("minimal", "default"):
        d = call(a.models.split(",")[0], condition, a.effort, "Reply with only the word: pong")
        r = record("check", a.models.split(",")[0], condition, a.effort, d)
        print(f"{condition:8s} result={r['text']!r} input_tokens={r['input_tokens']} "
              f"thinking={r['thinking_tokens']} turns={r['num_turns']} list_cost=${r['list_cost_usd']}")
    print("Expect (CLI 2.1.328): minimal ~750 input tokens, default ~22,000.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["check", "run"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--conditions", default="minimal,default")
    ap.add_argument("--n-per-genre", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="low", help="'' to use the CLI default")
    a = ap.parse_args()
    (check if a.action == "check" else run)(a)
