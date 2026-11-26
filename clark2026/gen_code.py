"""Coding setting of Clark et al. (2026), appendix "Code": agentic bug sweeps in Pi, one Docker container per session.

Per session: a fresh clark-pi:0.84.4 container; the repository cloned at the paper's commit and its dependencies
installed in /opt/venv before the agent starts. Pi runs with its default system prompt and tools, no extensions,
skills, prompt templates or context files, compaction and retries off; one `pi --mode json` process per episode,
resuming the same session file. The simulated user (Claude Sonnet 4.6, the paper's repo-owner persona) sees only the
text of the conversation. Stop after 11 episodes or once the episode-final text reaches ~8,000 tokens (checked between
episodes); then one fixed closing message asks for MEMORY.md (collected, left out of the analysis).
Unit of analysis: turn-final replies (the last text with no tool call after it).
Deviation: the paper's sandbox allowed only the API proxy; our container has normal network access (it must reach
NanoGPT). It cannot see the host's files or credentials.

usage (repo root): uv run python clark2026/gen_code.py [N_REPOS] model [model ...]   model = a key of PRICES; its route
(NanoGPT by default, OpenRouter or local Ollama) is in PROVIDER. Env: CODE_BUDGET (default 8), CODE_PAR (parallel sessions,
default 5), SIM_VIA=openrouter for the simulated user. Needs Docker and the image: docker build -t clark-pi:0.84.4 clark2026/code
Output: clark2026/out/code/<model>.jsonl (one session per line) + diffs; resumable. Own budget CODE_BUDGET.
"""
import glob, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from llm import NANO_KEY, KEY as OR_KEY  # noqa: E402
from gen_chat import simulated_user, with_retry  # noqa: E402

CODE_BUDGET = float(os.environ.get("CODE_BUDGET", 8.0))
OUT = os.path.join(HERE, "out", "code")
IMAGE = "clark-pi:0.84.4"
MAX_EPISODES, MAX_REPLY_TOKENS = 11, 8000
REPOS = [  # (id, repo, commit, language, domain, install, test command) - the paper's Table (b)
    ("c01", "alichtman/shallow-backup", "8fc8234", "Python",
     "command-line tool that backs up dotfiles, package lists and application configuration, with git integration",
     "pip install -q -e . -r requirements.txt", "python -m pytest tests -q"),
    ("c03", "funkybob/stencil", "1dd661b", "Python",
     "single-file Django-style text template engine", "pip install -q -e '.[test]' pytest", "python -m pytest tests -q"),
    ("c04", "fabiocaccamo/python-fsutil", "deb6684", "Python",
     "filesystem-operations library", "pip install -q -e . -r requirements-test.txt", "python -m pytest tests -q"),
    ("c05", "con2/emrichen", "1690fb9", "Python",
     "YAML/JSON template engine", "pip install -q -e '.[test]' pytest", "python -m pytest -q"),
    ("c06", "obijywk/grilops", "18a6641", "Python",
     "library for encoding grid logic puzzles as constraints for the Z3 SMT solver",
     "pip install -q -e . -r requirements.txt", "python test/golden.py"),
]
FIRST = ("I've got a small project checked out here ({name}, {language}). Can you sweep the codebase for bugs and propose "
         "fixes? Please actually make the fixes you're confident about. Use quick local checks ({test}) rather than "
         "anything heavyweight — no deployments, no installing big new dependencies.\n\nOne thing: this box has no "
         "network access, and the dependencies are already installed in {path}, so nothing needs fetching.")
PERSONA = ("the developer who owns this repository. You know the project's purpose and history but not the internals of "
           "every module — you are not an expert on this code. You ask about specific findings the assistant reports, "
           "ask it to explain fixes in plain terms, question changes that look risky, and sometimes ask for one more pass "
           "over an area you care about. You write like a working developer in a terminal: brief, concrete, lowercase is fine")
CLOSING = ("before we stop: please write a short summary of what you found and changed, and anything still open, to "
           "MEMORY.md in the repo root so we can pick this up again later.")
# NanoGPT prices ($/M input, output, cache read) so Pi can cost each message; reasoning per the paper (medium)
PRICES = {"openai/gpt-5.6-luna": (0.2, 1.2, 0.02), "z-ai/glm-5": (1.0, 3.2, 0.2), "moonshotai/kimi-k2.5": (0.557, 2.832, 0.0912),
          "qwen/qwen3.8-27b": (0.035, 1.5, 0.018), "mistralai/mistral-small-3.1-24b-instruct": (0.351, 0.555, 0.351)}
# GLM 5 through NanoGPT wrote tool calls as plain text (<use_tool ...>), ending turns early, so it and the niche models
# go through OpenRouter (prices above are OpenRouter's; Mistral has no cache discount there)
PRICES.update({  # NanoGPT finetunes (input, output, cache read per 1M; cache reads as quoted by NanoGPT)
    "z-ai/glm-5.3-flash-uncensored": (0.2, 0.8, 0.07), "deepseek/deepseek-v4.1-flash-uncensored": (0.33, 1.32, 0.00825),
    "Gemma-4-31B-Gembrain-uncensored-heretic": (0.306, 0.306, 0.05), "qwen/qwen3.8-27b-antislop": (0.1, 0.6, 0.05)})
PRICES["hemmingway-1-96k"] = (0, 0, 0)  # local Ollama: richardyoung/hemmingway-1:Q4_K_M with num_ctx 98304, 8-bit KV cache (OLLAMA_KV_CACHE_TYPE=q8_0, flash attention)
PROVIDER = {"hemmingway-1-96k": "ollama", "z-ai/glm-5": "openrouter", "qwen/qwen3.8-27b": "openrouter", "mistralai/mistral-small-3.1-24b-instruct": "openrouter"}
BASE = {"nanogpt": ("https://nano-gpt.com/api/v1", "NANOGPT_API_KEY", NANO_KEY), "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", OR_KEY),
        "ollama": ("http://host.docker.internal:11434/v1", "OLLAMA_API_KEY", "ollama")}
CONTEXT = {"hemmingway-1-96k": 98304}  # what fits fully on a 24 GB GPU next to the 17 GB weights with an 8-bit KV cache
THINKING = {"openai/gpt-5.6-luna": "medium", "z-ai/glm-5": "medium", "moonshotai/kimi-k2.5": "medium",
            "qwen/qwen3.8-27b": "off", "mistralai/mistral-small-3.1-24b-instruct": "off",
            "z-ai/glm-5.3-flash-uncensored": "low", "deepseek/deepseek-v4.1-flash-uncensored": "off",
            "Gemma-4-31B-Gembrain-uncensored-heretic": "off", "qwen/qwen3.8-27b-antislop": "off",
            "hemmingway-1-96k": "off"}  # as in essays/chat (GLM 5.3 Flash at low)
PI_FLAGS = ["--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files", "--no-approve"]
ENV = ["-e", "PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
       "-e", "PI_CODING_AGENT_DIR=/root/pi-config", "-e", "PI_OFFLINE=1", "-e", "PI_TELEMETRY=0",
       "-e", "PI_SKIP_VERSION_CHECK=1"]
lock = threading.Lock()


def sh(*args, check=True, timeout=1800, input=None):
    return subprocess.run(list(args), capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=check, timeout=timeout, input=input)


def spent():
    return sum(json.loads(l).get("cost", 0) for f in glob.glob(os.path.join(OUT, "*.jsonl")) for l in open(f, encoding="utf-8"))


def pi_config(model):
    inp, out, cache = PRICES[model]
    prov = PROVIDER.get(model, "nanogpt")
    url, env, _ = BASE[prov]
    models = {"providers": {prov: {
        "baseUrl": url, "api": "openai-completions", "apiKey": "$" + env,
        "models": [{"id": model, "reasoning": THINKING[model] != "off", "contextWindow": CONTEXT.get(model, 200000), "maxTokens": 32000 if model not in CONTEXT else 8192,
                    "cost": {"input": inp, "output": out, "cacheRead": cache, "cacheWrite": inp}}]}}}
    if prov == "ollama":  # Ollama thinks unless told reasoning_effort "none"; Pi sends thinkingLevelMap.off when thinking is off
        m = models["providers"][prov]["models"][0]
        m.update({"reasoning": True, "thinkingLevelMap": {"off": "none"},
                  "compat": {"supportsReasoningEffort": True, "supportsDeveloperRole": False}})
    settings = {"compaction": {"enabled": False}, "retry": {"enabled": False}, "defaultProjectTrust": "never"}
    return json.dumps(models), json.dumps(settings)


def start_container(name, repo):
    rid, slug, commit, _, _, install, test = repo
    sh("docker", "rm", "-f", name, check=False)
    sh("docker", "run", "-d", "--name", name, "--memory", "6g", IMAGE)
    d = slug.split("/")[1]
    setup = (f"set -e; git clone -q https://github.com/{slug}.git /workspace/{d} && cd /workspace/{d} && git checkout -q {commit} "
             f"&& python3 -m venv /opt/venv && . /opt/venv/bin/activate && pip install -q --upgrade pip && {install}")
    r = sh("docker", "exec", name, "bash", "-c", setup, check=False, timeout=1800)
    if r.returncode:
        raise RuntimeError(f"setup failed for {slug}: {r.stderr[-500:]}")
    return f"/workspace/{d}"


def episode(name, workdir, model, message):
    """One Pi process for one user message. Returns (turn-final reply or None, cost, assistant messages)."""
    prov = PROVIDER.get(model, "nanogpt")
    _, env, key = BASE[prov]
    cmd = ["docker", "exec", "-w", workdir, *ENV, "-e", f"{env}={key}", name,
           "pi", "--mode", "json", "--provider", prov, "--model", model, "--thinking", THINKING[model],
           "--session", "/root/sessions/session.jsonl", *PI_FLAGS, message]
    # a failed turn is retried from the same session state, so the user message is not stacked into the session again
    S = "/root/sessions/session.jsonl"
    sh("docker", "exec", name, "bash", "-c", f"if [ -f {S} ]; then cp {S} {S}.bak; else rm -f {S}.bak; fi", check=False)
    try:
        return _run_episode(cmd)
    except Exception:
        sh("docker", "exec", name, "bash", "-c", f"if [ -f {S}.bak ]; then cp {S}.bak {S}; else rm -f {S}; fi", check=False)
        raise


def _run_episode(cmd):
    r = sh(*cmd, check=False, timeout=3600)
    msgs = []
    for line in r.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "message_end" and (ev.get("message") or {}).get("role") == "assistant":
            msgs.append(ev["message"])
    if not msgs:
        raise RuntimeError(f"pi produced no assistant message: {(r.stderr or r.stdout)[-400:]}")
    cost = sum(((m.get("usage") or {}).get("cost") or {}).get("total", 0) or 0 for m in msgs)
    last = msgs[-1]
    if last.get("stopReason") == "error":
        raise RuntimeError(f"pi error: {last.get('errorMessage', '')[:300]}")
    blocks = last.get("content") or []
    reply = None
    if not any(b.get("type") == "toolCall" for b in blocks):  # an episode ending on a tool call has no reply
        reply = "\n\n".join(b["text"] for b in blocks if b.get("type") == "text").strip() or None
    return reply, cost, msgs


def patient(fn):
    """Provider overloads and timeouts (NanoGPT finetunes) back off for up to ~13 minutes before the session is lost;
    other errors keep the paper's three quick tries. Pi keeps the session file, so a retried turn continues it."""
    for wait in (30, 60, 120, 240, 300, None):
        try:
            return with_retry(fn)
        except Exception as e:
            if wait is None or not any(k in str(e).lower() for k in ("overloaded", "429", "timed out", "503", "unavailable")):
                raise
            print(f"    transient provider error, retrying in {wait}s: {str(e)[:80]}", flush=True)
            time.sleep(wait)


def session(model, repo):
    rid, slug, commit, language, domain, install, test = repo
    name = f"clark-{model.replace('/', '-').replace('.', '-')}-{rid}"
    workdir = start_container(name, repo)
    try:
        models_json, settings_json = pi_config(model)
        sh("docker", "exec", "-i", name, "bash", "-c", "mkdir -p /root/pi-config /root/sessions && cat > /root/pi-config/models.json",
           input=models_json)
        sh("docker", "exec", "-i", name, "bash", "-c", "cat > /root/pi-config/settings.json", input=settings_json)
        sim_seed = (rid, f"reviewing an automated bug sweep of {slug.split('/')[1]} ({language}, {domain})", None)
        msg = FIRST.format(name=slug.split("/")[1], language=language, test=test, path="/opt/venv")
        convo, episodes, cost_model, cost_user, reply_tokens, n_msgs = [], [], 0.0, 0.0, 0, 0
        while True:
            convo.append({"role": "user", "text": msg})
            reply, c, msgs = patient(lambda: episode(name, workdir, model, msg))
            cost_model += c
            n_msgs += len(msgs)
            episodes.append({"user": msg, "reply": reply, "assistant_messages": len(msgs), "cost": c})
            if reply:
                convo.append({"role": "assistant", "text": reply})
                reply_tokens += int(len(reply.split()) * 1.33)
            if len(episodes) >= MAX_EPISODES or reply_tokens >= MAX_REPLY_TOKENS:
                break
            msg = ""
            # no reply (turn ended on a tool call): the simulated user's view would end on its own message, which
            # OpenRouter treats as a prefill (400s, or Sonnet continuing the transcript), so use the plain fallback
            for _ in range(2 if reply else 0):
                msg, cu = with_retry(lambda: simulated_user(sim_seed, convo, persona=PERSONA))
                cost_user += cu
                if msg and len(msg) <= 2000:
                    break
            if not msg or len(msg) > 2000:
                msg = "ok, keep going"
        closing, c, _ = patient(lambda: episode(name, workdir, model, CLOSING))
        cost_model += c
        diff = sh("docker", "exec", "-w", workdir, name, "git", "diff", check=False).stdout
        return {"model": model, "repo": slug, "repo_id": rid, "commit": commit, "episodes": episodes, "closing_reply": closing,
                "turn_final_replies": [e["reply"] for e in episodes if e["reply"]], "reply_tokens_est": reply_tokens,
                "stop": "turns" if len(episodes) >= MAX_EPISODES else "tokens", "diff": diff, "assistant_messages": n_msgs,
                "cost_model": cost_model, "cost_user": cost_user, "cost": cost_model + cost_user}
    finally:
        sh("docker", "rm", "-f", name, check=False)


def main():
    n = int(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    state = {"spent": spent()}
    print(f"coding: already spent ${state['spent']:.2f} of ${CODE_BUDGET}", flush=True)
    for model in sys.argv[2:]:
        path = os.path.join(OUT, model.replace("/", "__") + ".jsonl")
        done = {json.loads(l)["repo_id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
        todo = [r for r in REPOS[:n] if r[0] not in done]

        def run(repo):
            if state["spent"] >= CODE_BUDGET:
                return
            try:
                rec = session(model, repo)
            except Exception as e:
                print(f"  {model} {repo[0]}: discarded after error: {e}", flush=True)  # the paper re-runs such sessions
                return
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                state["spent"] += rec["cost"]
            print(f"  {model} {repo[0]}: {len(rec['episodes'])} episodes, {len(rec['turn_final_replies'])} replies, "
                  f"stop={rec['stop']}, ${rec['cost']:.2f}", flush=True)
        with ThreadPoolExecutor(int(os.environ.get("CODE_PAR", 5))) as ex:  # fewer for models NanoGPT overloads
            list(ex.map(run, todo))
        print(f"{model}: spent ${state['spent']:.2f}", flush=True)
        if state["spent"] >= CODE_BUDGET:
            print("CODE_BUDGET reached")
            break


if __name__ == "__main__":
    main()
