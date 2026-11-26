"""Compare our paired Cohen's d (results/features_vs_human_paper.csv) with the paper's Table S6 (paper_table_s6.csv)."""
import re
import pandas as pd

NAME2F = {
    "Past tense": "f_01_past_tense", "Perfect aspect": "f_02_perfect_aspect", "Present tense": "f_03_present_tense",
    "Place adverbials": "f_04_place_adverbials", "Time adverbials": "f_05_time_adverbials",
    "First-person pronouns": "f_06_first_person_pronouns", "Second-person pronouns": "f_07_second_person_pronouns",
    "Third-person pronouns": "f_08_third_person_pronouns", "Pronoun ‘it’": "f_09_pronoun_it",
    "Demonstrative pronouns": "f_10_demonstrative_pronoun", "Indefinite pronouns": "f_11_indefinite_pronouns",
    "Pro-verb ‘do’": "f_12_proverb_do", "‘Wh-’ questions": "f_13_wh_question", "Nominalizations": "f_14_nominalizations",
    "Gerunds": "f_15_gerunds", "Other nouns": "f_16_other_nouns", "Agentless passives": "f_17_agentless_passives",
    "by-passives": "f_18_by_passives", "‘Be’ as main verb": "f_19_be_main_verb",
    "Existential ‘there’": "f_20_existential_there", "‘That’ verb complements": "f_21_that_verb_comp",
    "‘That’ adjective complements": "f_22_that_adj_comp", "‘Wh-’ clauses": "f_23_wh_clause",
    "Infinitives": "f_24_infinitives", "Present participial clauses": "f_25_present_participle",
    "Past participial clauses": "f_26_past_participle", "Past participial postnominal": "f_27_past_participle_whiz",
    "Present participial postnominal": "f_28_present_participle_whiz", "‘That’ clauses as subject": "f_29_that_subj",
    "‘That’ clauses as object": "f_30_that_obj", "‘Wh-’ relatives as subject": "f_31_wh_subj",
    "‘Wh-’ relatives as object": "f_32_wh_obj", "Pied-piping relative clauses": "f_33_pied_piping",
    "Sentence relatives": "f_34_sentence_relatives", "Because": "f_35_because", "Though, although": "f_36_though",
    "If, unless": "f_37_if", "Other adverbial subordinators": "f_38_other_adv_sub",
    "Prepositional phrases": "f_39_prepositions", "Attributive adjectives": "f_40_adj_attr",
    "Predicative adjectives": "f_41_adj_pred", "Adverbs": "f_42_adverbs", "Mean word length": "f_44_mean_word_length",
    "Conjuncts": "f_45_conjuncts", "Downtoners": "f_46_downtoners", "Hedges": "f_47_hedges",
    "Amplifiers": "f_48_amplifiers", "Emphatics": "f_49_emphatics", "Discourse particles": "f_50_discourse_particles",
    "Demonstratives": "f_51_demonstratives", "Possibility modals": "f_52_modal_possibility",
    "Necessity modals": "f_53_modal_necessity", "Predictive modals": "f_54_modal_predictive",
    "Public verbs": "f_55_verb_public", "Private verbs": "f_56_verb_private", "Suasive verbs": "f_57_verb_suasive",
    "‘Seem’ and ‘appear’": "f_58_verb_seem", "Contractions": "f_59_contractions", "‘That’ deletion": "f_60_that_deletion",
    "Stranded prepositions": "f_61_stranded_preposition", "Split infinitives": "f_62_split_infinitive",
    "Split auxiliaries": "f_63_split_auxiliary", "Phrasal co-ordination": "f_64_phrasal_coordination",
    "Clausal co-ordination": "f_65_clausal_coordination", "Synthetic negation": "f_66_neg_synthetic",
    "Analytic negation": "f_67_neg_analytic",
}
COLS = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "llama-3-8B-Instruct", "llama-3-70B-Instruct",
        "llama-3-8B", "llama-3-70B"]  # Table S6 column order

# Table S6 (paired Cohen's d, LLM vs human, per feature), transcribed from the paper's appendix
paper = pd.read_csv("paper_table_s6.csv").set_index("feature")[COLS].stack().rename("d_paper")
assert len(paper) == 66 * 6, len(paper)

ours = pd.read_csv("results/features_vs_human_paper.csv").set_index(["feature", "source"]).d.rename("d_ours")
m = pd.concat([paper, ours], axis=1, join="inner").reset_index(names=["feature", "source"])
print(f"{len(m)} feature x model cells; overall r = {m.d_paper.corr(m.d_ours):.3f}, "
      f"sign agreement (|d_paper| > 0.2): {(m[m.d_paper.abs() > .2].pipe(lambda x: (x.d_paper > 0) == (x.d_ours > 0))).mean():.1%}")
print(m.groupby("source").apply(lambda g: pd.Series({"r": g.d_paper.corr(g.d_ours),
                                                      "mean_abs_diff": (g.d_paper - g.d_ours).abs().mean()})).round(3))
m["diff"] = m.d_ours - m.d_paper
m.to_csv("results/replication_vs_paper.csv", index=False)
print("\nLargest disagreements:")
print(m.reindex(m["diff"].abs().sort_values(ascending=False).index).head(12).round(2).to_string(index=False))
