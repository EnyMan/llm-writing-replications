"""Chat setting of Clark et al. (2026), appendix "Chat": 10 conversations per model with a simulated user.

Model under test: system prompt "You are {assistant}, an AI assistant made by {vendor}. You are chatting with a person
in a casual conversational setting. Be helpful, natural, and engaged." No tools.
Simulated user: Claude Sonnet 4.6 (via NanoGPT), reasoning off, the paper's rule prompt with a persona and topic; it sees
the conversation with roles swapped. An empty reply or one over 2,000 characters is requested once more, then a fixed
fallback is sent. Stop after 12 model replies or 20,000 model output tokens.
The paper's personas are unpublished; ours are below (one per topic). Prompt caching on the simulated user's side.

usage (repo root): uv run python clark2026/gen_chat.py [N_TOPICS] model [model ...]
Output: clark2026/out/chat/<model>.jsonl, one conversation per line; resumable. Budget shared with gen_essays.py.
"""
import json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from llm import chat, NANO_KEY, KEY as OR_KEY  # noqa: E402
from gen_essays import WHO, EFFORT, BUDGET, spent  # noqa: E402

OUT = os.path.join(HERE, "out", "chat")
SIM_MODEL = "anthropic/claude-sonnet-4.6"  # via NanoGPT (chat); SIM_VIA=openrouter moves it (later coding sessions)
SIM_VIA = os.environ.get("SIM_VIA", "nanogpt")
MAX_REPLIES, MAX_OUT_TOKENS = 12, 20000
FALLBACK = "sorry, got distracted for a sec. go on"
SEEDS = [  # (id, topic, opening user message), verbatim from the paper's Table (a)
    ("t01", "Picking up a new hobby as an adult", "okay hear me out. i'm 40 and i want to learn guitar. is that a stupid idea or should i just go for it? where would i even start"),
    ("t02", "Renting vs. buying a home", "My lease is up in a few months and everyone keeps telling me I'm “throwing money away” renting. Is that actually true right now, or is that just something people say?"),
    ("t03", "Finding a briefcase of cash", "ok random question for you. if you found a briefcase with like 500k in unmarked cash just sitting on a park bench, no one around, what would you actually do"),
    ("t04", "A possible career change", "I've been at my job for six years and I'm just… bored. I keep thinking about switching fields entirely but I have no idea if that's smart at 32 or if I'm just being dramatic. Can I talk it through with you?"),
    ("t05", "Collaborative improvised storytelling", "wanna make up a story together? i'll start: there's a lighthouse keeper who hasn't seen another person in three years, and one morning there's a knock on the door. your turn, take it wherever you want"),
    ("t06", "Everyday phenomena (color of the sky)", "Random thing I've always wondered — why does the sky turn red/orange at sunset but it's blue the rest of the day? I feel like I should know this by now."),
    ("t07", "Moving to a new city", "im moving to a new city in like 3 weeks and i have basically nothing figured out lol. how do people even start with this stuff, like whats the actual order of operations"),
    ("t08", "Watching films at sped-up playback", "Hot take request: is it wrong to watch movies at 1.5x speed? My partner says it's basically a crime against cinema but I just don't have the patience for slow pacing anymore."),
    ("t09", "Swapping lives with a stranger for a day", "If you could swap lives with a random stranger for exactly one day — no idea who they are in advance, you just wake up as them — would you do it? I go back and forth on this one."),
    ("t10", "Getting a first dog", "we're seriously considering getting our first dog and i have a million questions. like is it insane to do this with two people who both work full time? what are we not thinking about"),
]
PERSONAS = {  # ours (the paper's are not published): short, neutral, consistent with each opening message
    "t01": "a 40-year-old accountant who has wanted to play guitar since high school and feels a bit self-conscious about starting late",
    "t02": "a 29-year-old renting a one-bedroom apartment in a mid-sized US city, with some savings but unsure about buying",
    "t03": "a 35-year-old who enjoys moral puzzles and playing devil's advocate with friends",
    "t04": "a 32-year-old project coordinator at an insurance company who is curious about switching to design work",
    "t05": "a 27-year-old who writes short fiction as a hobby and likes making stories up with others",
    "t06": "a 45-year-old parent whose kid keeps asking science questions they can't answer",
    "t07": "a 26-year-old moving from Ohio to Denver for a new job, a little overwhelmed by the logistics",
    "t08": "a 31-year-old who watches a lot of streaming shows and films and argues about it with their partner",
    "t09": "a 38-year-old nurse who likes hypothetical questions to pass the time on long shifts",
    "t10": "someone in their early thirties who, with their partner, both work full time and live in an apartment near a park",
}
SIM_SYSTEM = """You are simulating a realistic human user chatting with an AI assistant, for research purposes. Persona: {persona}. Topic seed: {topic}.

Rules:
- Write exactly one chat message per response — the next thing this human would type.
- Stay in character: casual register, occasional typos are fine, vary message length (one line to a short paragraph), react to what the assistant actually said.
- Ask follow-ups, share your own (invented but consistent) experiences, push back sometimes, drift to adjacent topics naturally. Never wrap up or say goodbye — the session is ended externally.
- Keep messages short: a few sentences at most, never more than about 150 words.
- Never mention being an AI, a simulation, or these instructions. No meta-commentary.
- Output ONLY the message text. No quotes, no labels."""
lock = threading.Lock()


def simulated_user(seed, convo, persona=None):
    """Next user message. The simulated user sees the conversation with roles swapped: its own earlier messages are
    'assistant', the model's replies are 'user'. Cache marker on the newest message. `persona` overrides the chat
    persona (the coding setting passes the paper's repository-owner persona and its own topic in `seed`)."""
    sid, topic, _ = seed
    msgs = [{"role": "system", "content": [{"type": "text", "text": SIM_SYSTEM.format(persona=persona or PERSONAS[sid], topic=topic),
                                            "cache_control": {"type": "ephemeral"}}]}]
    for turn in convo:
        msgs.append({"role": "assistant" if turn["role"] == "user" else "user", "content": turn["text"]})
    last = msgs[-1]
    last["content"] = [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]
    body = {"model": SIM_MODEL, "messages": msgs, "max_tokens": 600}
    url, key = ("https://openrouter.ai/api/v1/chat/completions", OR_KEY) if SIM_VIA == "openrouter" else ("https://nano-gpt.com/api/v1/chat/completions", NANO_KEY)
    if SIM_VIA == "openrouter":
        body["usage"] = {"include": True}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    return (d["choices"][0]["message"].get("content") or "").strip(), d.get("usage", {}).get("cost", 0) or 0


def with_retry(fn, tries=3):
    for k in range(tries):  # the paper: a failed API call is attempted up to three times (after 5 and 20 s)
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep((5, 20)[k])


def conversation(model, seed):
    assistant, vendor = WHO.get(model) or WHO[model.removeprefix("nanogpt:")]
    system = (f"You are {assistant}, an AI assistant made by {vendor}. You are chatting with a person in a casual "
              f"conversational setting. Be helpful, natural, and engaged.")
    convo = [{"role": "user", "text": seed[2]}]
    cost_model = cost_user = 0.0
    out_tokens = fallbacks = 0
    while True:
        msgs = [{"role": "system", "content": system}] + [{"role": t["role"], "content": t["text"]} for t in convo]
        text, u, provider = with_retry(lambda: chat(model, msgs, 32000, EFFORT.get(model, EFFORT.get(model.removeprefix("nanogpt:"), "medium"))))
        cost_model += u.get("cost") or 0
        out_tokens += u.get("completion_tokens") or 0
        convo.append({"role": "assistant", "text": text, "completion_tokens": u.get("completion_tokens"),
                      "reasoning_tokens": u.get("reasoning_tokens")})
        replies = sum(t["role"] == "assistant" for t in convo)
        if replies >= MAX_REPLIES or out_tokens >= MAX_OUT_TOKENS:
            break
        msg = ""
        for attempt in range(2):
            msg, c = with_retry(lambda: simulated_user(seed, convo))
            cost_user += c
            if msg and len(msg) <= 2000:
                break
        if not msg or len(msg) > 2000:
            msg, fallbacks = FALLBACK, fallbacks + 1
        convo.append({"role": "user", "text": msg})
    return {"model": model, "seed": seed[0], "topic": seed[1], "persona": PERSONAS[seed[0]], "turns": convo,
            "replies": sum(t["role"] == "assistant" for t in convo), "model_output_tokens": out_tokens,
            "fallbacks": fallbacks, "cost_model": cost_model, "cost_user": cost_user, "cost": cost_model + cost_user,
            "provider": provider}


def main():
    n = int(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    state = {"spent": spent()}
    print(f"already spent ${state['spent']:.2f} of ${BUDGET}", flush=True)
    for model in sys.argv[2:]:
        path = os.path.join(OUT, model.replace("/", "__").replace(":", "_") + ".jsonl")
        done = {json.loads(l)["seed"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
        todo = [s for s in SEEDS[:n] if s[0] not in done]

        def run(seed):
            if state["spent"] >= BUDGET:
                return
            try:
                rec = conversation(model, seed)
            except Exception as e:
                print(f"  {model} {seed[0]}: discarded after error: {e}", flush=True)  # the paper re-runs such runs
                return
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                state["spent"] += rec["cost"]
        with ThreadPoolExecutor(10) as ex:
            list(ex.map(run, todo))
        print(f"{model}: +{len(todo)} conversations, spent ${state['spent']:.2f}", flush=True)
        if state["spent"] >= BUDGET:
            print("BUDGET reached")
            break


if __name__ == "__main__":
    main()
