"""Biber features with settings that match the published HAP-E features (see validate_biber.py)."""
import pandas as pd
import polars as pl
import spacy
import pybiber

_nlp = None


def tag(df: pd.DataFrame) -> pd.DataFrame:
    """df[doc_id, text] (doc_id unique) -> features indexed by doc_id."""
    global _nlp
    _nlp = _nlp or spacy.load("en_core_web_sm", disable=["ner"])
    tokens = pybiber.CorpusProcessor().process_corpus(pl.from_pandas(df[["doc_id", "text"]]), nlp_model=_nlp)
    return pybiber.biber(tokens, force_ttr=True, strict_be_main_verb=False).to_pandas().set_index("doc_id")
