"""N-gram measures of Clark et al. (2026): unattested word pairs and trigram perplexity against a reference corpus.

The paper used a KenLM trigram (modified Kneser-Ney). This is a numpy re-implementation (interpolated modified KN,
Chen & Goodman discounts per order), built on lowercased word tokens with <s> </s> sentence boundaries.
ponytail: OOV words map to <unk> with a fixed unigram floor; KenLM's exact <unk> handling differs slightly.
"""
import re
import numpy as np

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STAND_IN = "zzname"


def tokens(sentence):
    return TOKEN.findall(sentence.lower().replace("’", "'"))


def _discounts(counts):
    n = np.bincount(np.minimum(counts, 4), minlength=5)[1:5].astype(float)  # n1..n4
    if (n == 0).any():  # tiny corpora: fall back to fixed discounts
        return np.array([0.0, 0.5, 1.0, 1.5])
    y = n[0] / (n[0] + 2 * n[1])
    d = np.array([0.0, 1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1], 3 - 4 * y * n[3] / n[2]])
    return np.clip(d, [0, 0.05, 0.05, 0.05], [0, 1, 2, 3])  # a zero discount would zero the backoff mass


class KN3:
    def __init__(self, sentences):
        vocab = {"<s>": 0, "</s>": 1, "<unk>": 2}
        seqs = []
        for s in sentences:
            ids = [0, 0] + [vocab.setdefault(w, len(vocab)) for w in tokens(s)] + [1]
            seqs.append(np.array(ids, dtype=np.int64))
        self.vocab, V = vocab, len(vocab)
        self.V = V
        a = np.concatenate([s[:-2] for s in seqs]); b = np.concatenate([s[1:-1] for s in seqs])
        c = np.concatenate([s[2:] for s in seqs])
        # trigram counts c(abc)
        self.k3, self.c3 = np.unique((a * V + b) * V + c, return_counts=True)
        # continuation counts for bigram (number of distinct left words) and unigram
        abc_b_c = self.k3 % (V * V)
        self.k2, self.cc2 = np.unique(abc_b_c, return_counts=True)  # N1+(. b c)
        bigram_c = self.k2 % V
        self.cc1 = np.bincount(bigram_c, minlength=V).astype(float)  # N1+(. . c) over bigram types
        self.p1 = (self.cc1 + 1e-9) / (self.cc1.sum() + V * 1e-9)
        self.p1[2] = max(self.p1[2], 1.0 / self.cc1.sum())  # <unk> floor
        self.p1 /= self.p1.sum()
        self.D3, self.D2 = _discounts(self.c3), _discounts(self.cc2)
        self._context_stats()
        self.bigram_set = np.unique(np.concatenate([s[1:-1] * V + s[2:] for s in seqs if len(s) > 3]))

    def _context_stats(self):
        V = self.V
        # trigram contexts (a,b)
        ctx3 = self.k3 // V
        self.k_ctx3, inv = np.unique(ctx3, return_inverse=True)
        self.tot3 = np.bincount(inv, weights=self.c3)
        self.gam3 = np.bincount(inv, weights=self.D3[np.minimum(self.c3, 3)])
        # bigram contexts (b) over continuation counts
        ctx2 = self.k2 // V
        self.k_ctx2, inv2 = np.unique(ctx2, return_inverse=True)
        self.tot2 = np.bincount(inv2, weights=self.cc2)
        self.gam2 = np.bincount(inv2, weights=self.D2[np.minimum(self.cc2, 3)])

    @staticmethod
    def _lookup(keys, vals, q, default=0.0):
        i = np.searchsorted(keys, q).clip(0, len(keys) - 1)
        hit = keys[i] == q
        return np.where(hit, vals[i] if np.ndim(vals) else vals, default), hit

    def ids(self, sentence):
        return np.array([0, 0] + [self.vocab.get(w, 2) for w in tokens(sentence)] + [1], dtype=np.int64)

    def logprobs(self, seq):
        """log10 P(w_i | w_{i-2} w_{i-1}) for every position i >= 2 of an id sequence."""
        return self.logprobs_abc(seq[:-2], seq[1:-1], seq[2:])

    def logprobs_abc(self, a, b, c):
        """Vectorized over any number of trigrams (a, b, c)."""
        V = self.V
        # bigram level (continuation counts), interpolated with unigram continuation
        cc2, _ = self._lookup(self.k2, self.cc2.astype(float), b * V + c)
        tot2, h2 = self._lookup(self.k_ctx2, self.tot2, b)
        gam2, _ = self._lookup(self.k_ctx2, self.gam2, b)
        p1 = self.p1[c]
        d2 = self.D2[np.minimum(cc2, 3).astype(int)]
        p2 = np.where(h2, (np.maximum(cc2 - d2, 0) + gam2 * p1) / np.where(h2, tot2, 1), p1)
        # trigram level
        c3, _ = self._lookup(self.k3, self.c3.astype(float), (a * V + b) * V + c)
        tot3, h3 = self._lookup(self.k_ctx3, self.tot3, a * V + b)
        gam3, _ = self._lookup(self.k_ctx3, self.gam3, a * V + b)
        d3 = self.D3[np.minimum(c3, 3).astype(int)]
        p3 = np.where(h3, (np.maximum(c3 - d3, 0) + gam3 * p2) / np.where(h3, tot3, 1), p2)
        return np.log10(np.maximum(p3, 1e-12))

    def score(self, sentences):
        """(unattested pairs per 1,000 pairs, perplexity per token incl. </s>), pooled over sentences.
        Pairs and n-grams containing the stand-in name are not counted."""
        V, stand = self.V, self.vocab.get(STAND_IN, -1)
        seqs = [q for q in map(self.ids, sentences) if len(q) > 3]  # one vectorized pass over all sentences
        a = np.concatenate([q[:-2] for q in seqs]); b = np.concatenate([q[1:-1] for q in seqs]); c = np.concatenate([q[2:] for q in seqs])
        lp = self.logprobs_abc(a, b, c)
        keep = (a != stand) & (b != stand) & (c != stand)
        lp_sum, n_tok = lp[keep].sum(), keep.sum()
        w1 = np.concatenate([q[2:-2] for q in seqs]); w2 = np.concatenate([q[3:-1] for q in seqs])  # adjacent word pairs
        ok = (w1 != stand) & (w2 != stand)
        qk = (w1 * V + w2)[ok]
        i = np.searchsorted(self.bigram_set, qk).clip(0, len(self.bigram_set) - 1)
        un, pairs = int((self.bigram_set[i] != qk).sum()), int(ok.sum())
        return 1000 * un / pairs, 10 ** (-lp_sum / n_tok)


if __name__ == "__main__":  # self-check: probabilities over the vocabulary sum to ~1 for a seen and an unseen context
    lm = KN3(["the cat sat on the mat", "the dog sat on the log", "a cat ate the fish"] * 3)
    V = lm.V
    for ctx in ([0, lm.vocab["the"]], [lm.vocab["mat"], lm.vocab["on"]]):
        tot = sum(10 ** lm.logprobs(np.array(ctx + [w]))[0] for w in range(V))
        assert abs(tot - 1) < 1e-6, tot
    u, ppl = lm.score(["the cat sat on the log"])
    assert u == 0 and ppl > 1
    u2, ppl2 = lm.score(["fish the on sat log cat"])
    assert u2 > 0 and ppl2 > ppl
    print("ok", round(ppl, 2), round(ppl2, 2))
