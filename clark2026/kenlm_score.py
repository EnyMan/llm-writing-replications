"""Score tokenized sources against the KenLM Wikipedia trigram (runs inside WSL: python3 + kenlm module).

Input: clark2026/out/ngram_inputs/<source>.txt, one tokenized sentence per line (written by ngram_run.py export).
Perplexity: per token including </s> (KenLM convention). Unattested pairs: adjacent word pairs whose bigram is not
in the model (matched n-gram length < 2), per 1,000 pairs. Pairs / tokens touching the stand-in name are skipped.
usage: python3 kenlm_score.py MODEL.binary INPUT_DIR OUT.csv
"""
import csv, glob, os, sys
import kenlm

SKIP = {"zzname", "zzcode"}  # stand-in name and code placeholder: n-grams containing them are not counted
model = kenlm.Model(sys.argv[1])
rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[2], "*.txt"))):
    lp, ntok, un, pairs = 0.0, 0, 0, 0
    for line in open(path, encoding="utf-8"):
        w = line.split()
        if not w:
            continue
        scores = list(model.full_scores(" ".join(w), bos=True, eos=True))  # len(w) + 1 entries (last is </s>)
        seq = ["<s>", "<s>"] + w + ["</s>"]
        for i, (logp, length, _oov) in enumerate(scores):
            if not SKIP.intersection(seq[i:i + 3]):
                lp += logp
                ntok += 1
            if 1 <= i < len(w) and not SKIP.intersection((w[i - 1], w[i])):
                pairs += 1
                un += length < 2
    rows.append({"source": os.path.basename(path)[:-4], "unattested": 1000 * un / pairs,
                 "perplexity": 10 ** (-lp / ntok), "tokens": ntok, "pairs": pairs})
    print(rows[-1], flush=True)
with open(sys.argv[3], "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
