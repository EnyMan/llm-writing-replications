"""Download the third-party data for the Clark et al. (2026) replication into clark2026/data and clark2026/models.

  persuade      PERSUADE 2.0 (Crossley et al. 2024, CC BY 4.0) train + test from the authors' Google Drive links
                (needs `gdown`; the test zip's password is published in their README), then the human baseline:
                independent assignments, holistic score 5-6, grade, ELL status and score present -> 2,934 essays
                (data/persuade_indep_score56.parquet) and the seven assignment texts (data/assignments.json)
  wikipedia     wikimedia/wikipedia 20231101.en, all 41 shards (~20 GB): the essay n-gram reference and the
                Wikipedia comparison sample
  opensubtitles OPUS OpenSubtitles v2018 English monolingual text (3.7 GB): the chat human corpus and n-gram reference
  stackexchange the StackExchange subset for the coding setting (~10 GB; runs get_stackexchange.py)
  torchcast     torchcast-ai/torchcast-decision-4b v1.3.1 (~10 GB), the local yes/no judge for the two
                classifier-based measures
usage (repo root): uv run python clark2026/get_data.py persuade [wikipedia ...] | all
"""
import os, subprocess, sys, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PERSUADE = {"persuade_2.0_train.csv": "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135",  # Google Drive ids from the corpus README
            "persuade_test.zip": "1K1SIJiG-2zWgMlTzxQeYOcLwOsFaVel1"}


def persuade():
    import gdown
    import pandas as pd
    for name, fid in PERSUADE.items():
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            gdown.download(id=fid, output=path)
    test = os.path.join(DATA, "persuade_corpus_2.0_test.csv")
    if not os.path.exists(test):
        with zipfile.ZipFile(os.path.join(DATA, "persuade_test.zip")) as z:
            z.extractall(DATA, pwd=b"persuade_test")  # 7-Zip works too if this zip method is not supported
    cols = ["essay_id", "full_text", "holistic_essay_score", "task", "prompt_name", "assignment", "grade_level", "ell_status"]
    tr = pd.read_csv(os.path.join(DATA, "persuade_2.0_train.csv"), low_memory=False, usecols=cols).drop_duplicates("essay_id")
    te = pd.read_csv(test, low_memory=False)
    te = te[[c for c in cols if c in te.columns]].drop_duplicates("essay_id")
    d = pd.concat([tr.assign(split="train"), te.assign(split="test")]).drop_duplicates("essay_id")
    e = d[d.task == "Independent"].dropna(subset=["grade_level", "ell_status", "holistic_essay_score"])
    k = e[e.holistic_essay_score >= 5]
    assert len(k) == 2934, f"expected the paper's 2,934 essays, got {len(k)}"
    k.to_parquet(os.path.join(DATA, "persuade_indep_score56.parquet"), index=False)
    print(f"PERSUADE human baseline: {len(k)} essays over {k.prompt_name.nunique()} assignments")
    # data/assignments.json (the seven assignment texts, in the order the essays were generated) ships with the repo


def wikipedia():
    from huggingface_hub import snapshot_download
    snapshot_download("wikimedia/wikipedia", repo_type="dataset", allow_patterns=["20231101.en/*"],
                      local_dir=os.path.join(DATA, "wikipedia"), max_workers=4)


def opensubtitles():
    path = os.path.join(DATA, "opensubtitles", "en.txt.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve("https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz", path)
    print("OpenSubtitles:", path)


def stackexchange():
    subprocess.run([sys.executable, os.path.join(HERE, "get_stackexchange.py")], check=True)


def torchcast():
    from huggingface_hub import snapshot_download
    snapshot_download("torchcast-ai/torchcast-decision-4b", revision="v1.3.1",
                      local_dir=os.path.join(HERE, "models", "torchcast-decision-4b"), max_workers=4)


if __name__ == "__main__":
    steps = {"persuade": persuade, "wikipedia": wikipedia, "opensubtitles": opensubtitles,
             "stackexchange": stackexchange, "torchcast": torchcast}
    os.makedirs(DATA, exist_ok=True)
    for name in (list(steps) if sys.argv[1:] == ["all"] else sys.argv[1:]) or sys.exit(__doc__):
        print(f"== {name}", flush=True)
        steps[name]()
