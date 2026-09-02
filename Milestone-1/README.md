# Milestone 1: Prototype
Disaster Tweet Classification

## Problem
Binary text classification: given a tweet, predict whether it describes a
real disaster (1) or not (0). Real-world use case: emergency services and
news organizations monitoring Twitter/X need to filter genuine disaster
reports from unrelated tweets that just use disaster-like language
(e.g. "this concert was ON FIRE").

## Dataset
[Kaggle: Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
(`nlp-getting-started/`)

- `train.csv` — 7,613 tweets with columns `id`, `keyword`, `location`, `text`, `target`
- `test.csv` — 3,263 tweets to predict (no labels)
- `sample_submission.csv` — Kaggle submission format

## Approach
- EDA: class balance, text/char length distributions, URL/mention/hashtag
  features, keyword frequency by class.
- Baseline model: TF-IDF (max 5,000 features, unigrams+bigrams) +
  Logistic Regression.
- Compared against a few-shot LLM classifier (Ollama) on a sample for
  reference.

## Demo
<video src="https://github.com/user-attachments/assets/802ae81b-e48a-428d-9098-a01573899e57" controls></video>
