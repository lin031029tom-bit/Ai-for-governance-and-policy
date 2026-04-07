from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILES = [
    BASE_DIR / "19Jan25_firstdatarev.json",
    BASE_DIR / "27Jan25_query_checked.json",
]
OUTPUT_DATASET = BASE_DIR / "merged_dataset.csv"
OUTPUT_RESULTS = BASE_DIR / "replication_results.json"
OUTPUT_SUMMARY = BASE_DIR / "replication_summary.txt"


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def doccano_to_sentence_labels(records: Sequence[dict]) -> Tuple[List[str], List[str]]:
    """Extract sentence-label pairs, excluding empty and 'unsure' labels."""
    sentences: List[str] = []
    labels: List[str] = []
    for entry in records:
        raw_labels = entry.get("label", [])
        if raw_labels and raw_labels[0].lower() != "unsure":
            sentences.append(entry["text"])
            labels.append(raw_labels[0])
    return sentences, labels


def greedy_deduplicate(
    sentences: Sequence[str], labels: Sequence[str], threshold: int = 90
) -> Tuple[List[str], List[str]]:
    """
    Greedy near-duplicate removal modeled on the repository's use of rapidfuzz.
    The first sentence in each similarity group is retained.
    """
    kept_sentences: List[str] = []
    kept_labels: List[str] = []
    used_indices = set()

    for i, source_sentence in enumerate(sentences):
        if i in used_indices:
            continue
        used_indices.add(i)
        kept_sentences.append(source_sentence)
        kept_labels.append(labels[i])

        for j in range(i + 1, len(sentences)):
            if j in used_indices:
                continue
            if fuzz.ratio(source_sentence, sentences[j]) >= threshold:
                used_indices.add(j)

    return kept_sentences, kept_labels


def build_dataframe(sentences: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame({"text": list(sentences), "label": list(labels)})
    df["binary_label"] = np.where(df["label"] == "Non-Incentive", "Non-Incentive", "Incentive")
    return df


def evaluate_task(
    texts: Sequence[str],
    labels: Sequence[str],
    task_name: str,
    seeds: Iterable[int] = range(10),
) -> Dict[str, object]:
    """
    Evaluate a lightweight offline baseline across 10 stratified 60/20/20 splits.
    The dev split is retained for design consistency with the paper, but is not used
    for hyperparameter tuning in this simplified replication.
    """
    weighted_f1s: List[float] = []
    macro_f1s: List[float] = []
    reports: List[dict] = []

    for seed in seeds:
        x_train, x_temp, y_train, y_temp = train_test_split(
            list(texts),
            list(labels),
            test_size=0.4,
            random_state=seed,
            stratify=list(labels),
        )
        x_dev, x_test, y_dev, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,
            random_state=seed,
            stratify=y_temp,
        )

        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 1),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LinearSVC(C=1.0, class_weight="balanced"),
                ),
            ]
        )

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        weighted_f1s.append(f1_score(y_test, predictions, average="weighted"))
        macro_f1s.append(f1_score(y_test, predictions, average="macro"))
        reports.append(classification_report(y_test, predictions, output_dict=True, zero_division=0))

    labelwise_f1: Dict[str, List[float]] = defaultdict(list)
    for report in reports:
        for label_name, values in report.items():
            if label_name in {"accuracy", "macro avg", "weighted avg"}:
                continue
            labelwise_f1[label_name].append(values["f1-score"])

    return {
        "task": task_name,
        "n_examples": len(texts),
        "weighted_f1_mean": float(np.mean(weighted_f1s)),
        "weighted_f1_sd": float(np.std(weighted_f1s)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_sd": float(np.std(macro_f1s)),
        "label_f1_mean": {k: float(np.mean(v)) for k, v in sorted(labelwise_f1.items())},
    }


def main() -> None:
    # 1) Load the two public input datasets from the GitHub repository.
    original_records = load_json(INPUT_FILES[0])
    hitl_records = load_json(INPUT_FILES[1])

    # 2) Convert Doccano-style records to sentence/label pairs.
    sents_a, labs_a = doccano_to_sentence_labels(original_records)
    sents_b, labs_b = doccano_to_sentence_labels(hitl_records)

    # 3) Merge and deduplicate using the same threshold logic documented in the repo.
    merged_sentences, merged_labels = greedy_deduplicate(sents_a + sents_b, labs_a + labs_b, threshold=90)
    df = build_dataframe(merged_sentences, merged_labels)
    df.to_csv(OUTPUT_DATASET, index=False)

    # 4) Prepare binary and multiclass tasks.
    binary_result = evaluate_task(
        texts=df["text"].tolist(),
        labels=df["binary_label"].tolist(),
        task_name="binary",
    )

    multiclass_df = df[df["label"] != "Non-Incentive"].copy()
    multiclass_result = evaluate_task(
        texts=multiclass_df["text"].tolist(),
        labels=multiclass_df["label"].tolist(),
        task_name="multiclass",
    )

    # 5) Save structured outputs.
    dataset_summary = {
        "rows_after_deduplication": int(len(df)),
        "binary_counts": Counter(df["binary_label"]).most_common(),
        "multiclass_counts": Counter(multiclass_df["label"]).most_common(),
        "all_label_counts": Counter(df["label"]).most_common(),
    }

    results = {
        "dataset_summary": dataset_summary,
        "binary": binary_result,
        "multiclass": multiclass_result,
        "notes": [
            "This script uses the authors' public GitHub input JSON files.",
            "It reproduces the 1,419-row merged dataset size reported in the paper,",
            "but the class counts reconstructed from the repository inputs differ slightly",
            "from the workshop paper's final counts (269 incentives here vs. 263 in the paper).",
            "Modeling is a lightweight TF-IDF + LinearSVC baseline, not the paper's transformer-embedding pipeline.",
        ],
    }

    with OUTPUT_RESULTS.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "Lightweight Replication Summary",
        "=" * 29,
        f"Rows after deduplication: {len(df)}",
        f"Binary counts: {dict(Counter(df['binary_label']))}",
        f"All label counts: {dict(Counter(df['label']))}",
        "",
        (
            "Binary weighted F1 (mean ± sd): "
            f"{binary_result['weighted_f1_mean'] * 100:.1f} ± {binary_result['weighted_f1_sd'] * 100:.1f}"
        ),
        (
            "Binary macro F1 (mean ± sd): "
            f"{binary_result['macro_f1_mean'] * 100:.1f} ± {binary_result['macro_f1_sd'] * 100:.1f}"
        ),
        (
            "Multiclass weighted F1 (mean ± sd): "
            f"{multiclass_result['weighted_f1_mean'] * 100:.1f} ± {multiclass_result['weighted_f1_sd'] * 100:.1f}"
        ),
        (
            "Multiclass macro F1 (mean ± sd): "
            f"{multiclass_result['macro_f1_mean'] * 100:.1f} ± {multiclass_result['macro_f1_sd'] * 100:.1f}"
        ),
        "",
        "Note: this is a lightweight offline baseline, not the paper's full transformer+SVM pipeline.",
    ]

    OUTPUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
