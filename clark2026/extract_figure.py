"""Recover plotted values from the paper's vector figure combined_compact.pdf (Fig. "Comparison of linguistic features").

Rows: chat, coding, essays. Panels (left to right): ORC with relativizer, ORC no relativizer (per 1k words),
denominal verbs, inanimate subjects (per 1k sentences), dependency length (std.), stacked nouns (per 1k words),
unattested pairs (per 1k pairs), trigram perplexity.
Diamonds -> model values; vertical dashed / dotted lines -> human baseline / Wikipedia.
Output: clark2026/paper_values.csv
"""
import os, re
import pandas as pd
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "..", "2610.29162v1", "combined_compact.pdf")
PANELS = ["orc_rel", "orc_norel", "denominal", "inanimate", "dep_len", "stacked", "unattested", "perplexity"]
SETTINGS = ["chat", "coding", "essay"]

page = pymupdf.open(PDF)[0]
words = [(round((w[0] + w[2]) / 2, 1), round((w[1] + w[3]) / 2, 1), w[4]) for w in page.get_text("words")]
num = re.compile(r"^-?\d*\.?\d+$")

# tick rows: y values where many numeric labels share a line
ys = pd.Series([y for x, y, t in words if num.match(t) and x > 50]).value_counts()
tick_ys = sorted(ys[ys >= 10].index)[:3]
assert len(tick_ys) == 3, tick_ys

rows = []
for si, ty in enumerate(tick_ys):
    top = tick_ys[si - 1] + 5 if si else 0
    # model labels: words left of the plots, between the previous tick row and this one; join words on one line
    labels = {}
    for x, y, t in words:
        if x < 52 and top < y < ty - 3 and not num.match(t) or (x < 52 and top < y < ty - 3 and t in ("5", "5.5")):
            labels.setdefault(y, []).append((x, t))
    labels = {y: " ".join(t for _, t in sorted(v)) for y, v in labels.items()}
    # ticks -> panels: a new panel starts when the value drops or jumps by >5x
    ticks = sorted((x, float(t)) for x, y, t in words if abs(y - ty) < 1 and num.match(t) and x > 50)
    panels, cur = [], []
    for x, v in ticks:
        if cur and (v <= cur[-1][1] or (cur[-1][1] > 0 and v > 5 * cur[-1][1])):
            panels.append(cur)
            cur = []
        cur.append((x, v))
    panels.append(cur)
    assert len(panels) == 8, (SETTINGS[si], panels)
    bounds = [(panels[i][-1][0] + panels[i + 1][0][0]) / 2 for i in range(7)]

    def panel_of(x):
        return sum(x > b for b in bounds)

    def value(pi, x):
        (x0, v0), (x1, v1) = panels[pi][0], panels[pi][-1]
        return v0 + (x - x0) * (v1 - v0) / (x1 - x0)

    for d in page.get_drawings():
        r = d["rect"]
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if not (top < cy < ty - 3):
            continue
        if d["type"] == "fs" and len(d["items"]) == 4 and r.width < 6:  # diamond
            ly = min(labels, key=lambda y: abs(y - cy))
            if abs(ly - cy) < 2.5:
                pi = panel_of(cx)
                rows.append((SETTINGS[si], PANELS[pi], labels[ly], round(value(pi, cx), 4)))
        elif d["type"] == "s" and r.width < 0.5 and r.height > 20 and d.get("dashes") not in (None, "[] 0"):
            pi = panel_of(cx)  # vertical reference line: dashed = human corpus, dotted = Wikipedia
            kind = "Wikipedia" if re.search(r"\[\s*[\d.]+\s+[\d.]+\s*\]", d["dashes"]) and float(
                re.findall(r"[\d.]+", d["dashes"])[0]) < 1.5 else "human"
            rows.append((SETTINGS[si], PANELS[pi], f"[{kind}]", round(value(pi, cx), 4)))

out = pd.DataFrame(rows, columns=["setting", "measure", "model", "value"]).drop_duplicates()
out.to_csv(os.path.join(HERE, "paper_values.csv"), index=False)
print(out.groupby("setting").size())
print(out[out.setting == "essay"].pivot_table(index="model", columns="measure", values="value", aggfunc="first")[PANELS].round(2).to_string())
