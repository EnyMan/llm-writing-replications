"""Build the GitHub Pages site in docs/ from each replication's report fragment and data.

  reinhart2025/report/index.html + report/*.json  -> docs/reinhart2025/index.html + docs/reinhart2025/data/*.json
  clark2026/report/index.html    + report/*.json  -> docs/clark2026/index.html    + docs/clark2026/data/*.json
  site/index.html (landing page) and site/assets/ -> docs/
The fragments are plain HTML + d3 scripts written against a small `dash` API (site/assets/dash.js).
usage (repo root): uv run python site/build.py     (after reinhart2025/build_report_data.py / clark2026/build_report.py)
"""
import os, re, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE, DOCS = os.path.join(ROOT, "site"), os.path.join(ROOT, "docs")
REPO = "https://github.com/EnyMan/llm-writing-replications"
D3 = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"
PAGES = {  # folder -> (page title, datasets the fragment reads, one-line description)
    "reinhart2025": ("Do LLMs Write Like Humans? A 2026 Replication", ["models", "features", "pairs", "replication", "vocab"],
                     "Reinhart et al. (2025) replicated on HAP-E and extended to 25 current models."),
    "clark2026": ("Why Is AI Text Hard to Read? A Replication", ["values", "agreement"],
                  "Clark, Martin & Mahowald (2026) replicated across essays, chat and coding."),
}
NAV = ('<nav class="site-nav"><a class="home" href="{up}">LLM writing replications</a>'
       '<a href="{up}reinhart2025/"{a1}>Do LLMs write like humans?</a>'
       '<a href="{up}clark2026/"{a2}>Why is AI text hard to read?</a><a href="' + REPO + '">Code and data</a></nav>')
FOOT = ('<footer class="site-foot"><p>Independent replications, not affiliated with the original authors. '
        'Code MIT, generated data and text CC BY 4.0; third-party corpora keep their own licences.</p>'
        '<p>Built with the help of AI: code, analyses and these pages were written largely by Claude Code (Anthropic) '
        'under the author\'s direction and review.</p>'
        '<p><a href="' + REPO + '">Source, data and reproduction guide on GitHub</a></p></footer>')


def page(title, desc, body, up, current="", datasets=()):
    nav = NAV.format(up=up, a1=' aria-current="page"' if current == "reinhart2025" else "",
                     a2=' aria-current="page"' if current == "clark2026" else "")
    scripts = f'<script src="{D3}"></script>\n<script src="{up}assets/dash.js"></script>\n' if datasets else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="{up}assets/site.css">
{scripts}</head>
<body data-datasets="{' '.join(datasets)}">
{nav}
<main class="site-main">
{body}
</main>
{FOOT}
</body>
</html>
"""


def main():
    if os.path.exists(DOCS):
        shutil.rmtree(DOCS)
    shutil.copytree(os.path.join(SITE, "assets"), os.path.join(DOCS, "assets"))
    landing = open(os.path.join(SITE, "index.html"), encoding="utf-8").read()
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8", newline=chr(10)) as f:
        f.write(page("LLM Writing Replications", "Two replications of studies on how AI-written text differs from human writing.",
                     landing, ""))
    for folder, (title, datasets, desc) in PAGES.items():
        frag = open(os.path.join(ROOT, folder, "report", "index.html"), encoding="utf-8").read()
        frag = re.sub(r'<meta name="dash-one-writer">\s*', "", frag)
        out = os.path.join(DOCS, folder)
        os.makedirs(os.path.join(out, "data"))
        for d in datasets:
            shutil.copy(os.path.join(ROOT, folder, "report", d + ".json"), os.path.join(out, "data", d + ".json"))
        with open(os.path.join(out, "index.html"), "w", encoding="utf-8", newline=chr(10)) as f:
            f.write(page(title, desc, frag, "../", folder, datasets))
    open(os.path.join(DOCS, ".nojekyll"), "w").close()  # serve files as they are
    print("built", DOCS)


if __name__ == "__main__":
    main()
