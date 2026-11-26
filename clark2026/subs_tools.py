"""OpenSubtitles helpers (stdlib only, runs in WSL next to KenLM).

  sample   one streaming pass over en.txt.gz: keep each line with probability P (seeded), stop at ~TARGET words ->
           sample.txt, one subtitle line per line. P is set so the sample spreads over the whole file, not its start.
  tokens   stdin lines -> lowercased word tokens (same regex as ngrams.tokens), one line per input line, for lmplz.
  heldout  IN.gz -> tokens of every line `sample` did not draw (the chat reference without its human sample)
usage: python3 subs_tools.py sample IN.gz OUT.txt | heldout IN.gz | python3 subs_tools.py tokens < text
"""
import gzip, random, re, sys

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokens(line):
    return TOKEN.findall(line.lower().replace("’", "'"))


if sys.argv[1] == "tokens":
    out = sys.stdout
    for line in sys.stdin:
        t = tokens(line)
        if t:
            out.write(" ".join(t) + "\n")
elif sys.argv[1] == "heldout":  # tokens of every line the seeded sample did NOT draw: the human sample stays out of its own reference
    rng, out = random.Random(0), sys.stdout
    with gzip.open(sys.argv[2], "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if rng.random() < 1 / 4500:  # same draw sequence as `sample`
                continue
            t = tokens(line)
            if t:
                out.write(" ".join(t) + "\n")
elif sys.argv[1] == "sample":
    TARGET, P = 500_000, 1 / 4500  # ~2.6B words in v2018 English -> ~580k words expected over the whole file
    rng = random.Random(0)
    kept, words = [], 0
    with gzip.open(sys.argv[2], "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if rng.random() < P:
                line = line.strip()
                if line:
                    kept.append(line)
                    words += len(line.split())
    rng.shuffle(kept)
    out, n = [], 0
    for line in kept:  # trim to the target after the full pass, so no part of the file is favoured
        out.append(line)
        n += len(line.split())
        if n >= TARGET:
            break
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"kept {len(kept)} lines / {words} words over the whole file; wrote {len(out)} lines / {n} words")
