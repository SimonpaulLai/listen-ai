"""
Evaluation script for NLP sentiment analysis algorithms.

Usage:
    python evaluate.py                         # use built-in labeled_sample.json
    python evaluate.py --data path/to/data.json
    python evaluate.py --label-with-llm --csv ../data/posts.csv  # label with Claude API
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── import classifiers from app.py ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from app import classify_text_lexicon  # noqa: E402


def classify_text_bert_eval(text: str, pipeline) -> str:
    THRESHOLD = 0.70
    result = pipeline(text, truncation=True, max_length=512)[0]
    if result["score"] < THRESHOLD:
        return "neutral"
    return result["label"].lower()  # "positive", "neutral", "negative"


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = ["positive", "neutral", "negative"]
    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / total if total else 0.0

    f1_per_class = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_per_class[label] = (2 * precision * recall / (precision + recall)
                               if (precision + recall) else 0.0)

    macro_f1 = sum(f1_per_class.values()) / len(labels)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "f1_per_class": f1_per_class}


# ── LLM labeling with Claude API ─────────────────────────────────────────────

def label_with_claude(texts: list[str], batch_size: int = 20) -> list[str]:
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    labels: list[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
        prompt = (
            "以下是一批中文社群媒體貼文。請對每一則貼文進行情感分類，"
            "只能回答 positive、negative 或 neutral 三個標籤之一。\n"
            "請以 JSON 陣列格式回覆，例如：[\"positive\", \"negative\", \"neutral\"]\n\n"
            f"貼文：\n{numbered}"
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # extract JSON array
        start = raw.find("[")
        end = raw.rfind("]") + 1
        batch_labels = json.loads(raw[start:end])
        labels.extend(batch_labels)
        print(f"  Labeled batch {i // batch_size + 1} ({len(labels)}/{len(texts)})")
        time.sleep(0.5)

    return labels


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifiers")
    parser.add_argument("--data", default="labeled_sample.json",
                        help="Path to JSON file with [{text, label}] entries")
    parser.add_argument("--label-with-llm", action="store_true",
                        help="Use Claude API to label texts (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--csv", default=None,
                        help="Path to posts.csv to label with LLM (used with --label-with-llm)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max samples to evaluate (default: 200)")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Skip BERT evaluation (faster, no GPU/transformers needed)")
    args = parser.parse_args()

    # ── load data ────────────────────────────────────────────────────────────
    if args.label_with_llm and args.csv:
        import csv
        print(f"Loading posts from {args.csv} ...")
        rows = []
        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row["content"])
                if len(rows) >= args.limit:
                    break
        print(f"Labeling {len(rows)} posts with Claude API ...")
        labels = label_with_claude(rows)
        samples = [{"text": t, "label": l} for t, l in zip(rows, labels)]
        out_path = "llm_labeled.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"Saved LLM-labeled data to {out_path}")
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"ERROR: {data_path} not found.")
            sys.exit(1)
        with open(data_path, encoding="utf-8") as f:
            samples = json.load(f)

    samples = samples[: args.limit]
    texts = [s["text"] for s in samples]
    y_true = [s["label"] for s in samples]

    print(f"\n{'='*60}")
    print(f"Dataset: {len(texts)} samples")
    label_dist = {l: y_true.count(l) for l in ["positive", "neutral", "negative"]}
    print(f"Distribution: {label_dist}")
    print("=" * 60)

    # ── Lexicon evaluation ───────────────────────────────────────────────────
    print("\n[1/2] Evaluating Lexicon-based classifier ...")
    t0 = time.perf_counter()
    y_pred_lexicon = [classify_text_lexicon(t)[0] for t in texts]
    lexicon_time = time.perf_counter() - t0
    lexicon_metrics = compute_metrics(y_true, y_pred_lexicon)

    # ── BERT evaluation ──────────────────────────────────────────────────────
    bert_metrics = None
    bert_time = None
    if not args.skip_bert:
        print("\n[2/2] Loading BERT model ...")
        try:
            from transformers import pipeline as hf_pipeline
            bert_pipe = hf_pipeline(
                "sentiment-analysis",
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                device=-1,
            )
            print("      Evaluating BERT classifier ...")
            t0 = time.perf_counter()
            y_pred_bert = [classify_text_bert_eval(t, bert_pipe) for t in texts]
            bert_time = time.perf_counter() - t0
            bert_metrics = compute_metrics(y_true, y_pred_bert)
        except Exception as e:
            print(f"      BERT evaluation skipped: {e}")
    else:
        print("\n[2/2] BERT evaluation skipped (--skip-bert).")

    # ── Results table ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS")
    print("=" * 60)

    header = f"{'Metric':<25} {'Lexicon':>12}"
    if bert_metrics:
        header += f" {'BERT':>12}"
    print(header)
    print("-" * (37 + (13 if bert_metrics else 0)))

    def row(name: str, lv: float, bv) -> str:
        line = f"{name:<25} {lv:>11.4f}"
        if bv is not None:
            line += f" {bv:>11.4f}"
        return line

    print(row("Accuracy", lexicon_metrics["accuracy"],
              bert_metrics["accuracy"] if bert_metrics else None))
    print(row("Macro F1", lexicon_metrics["macro_f1"],
              bert_metrics["macro_f1"] if bert_metrics else None))
    for label in ["positive", "neutral", "negative"]:
        lv = lexicon_metrics["f1_per_class"][label]
        bv = bert_metrics["f1_per_class"][label] if bert_metrics else None
        print(row(f"  F1 ({label})", lv, bv))

    print("-" * (37 + (13 if bert_metrics else 0)))
    print(f"{'Inference time (s)':<25} {lexicon_time:>11.4f}", end="")
    if bert_time is not None:
        print(f" {bert_time:>11.4f}", end="")
    print()
    print(f"{'Time per sample (ms)':<25} {lexicon_time/len(texts)*1000:>11.4f}", end="")
    if bert_time is not None:
        print(f" {bert_time/len(texts)*1000:>11.4f}", end="")
    print()
    print("=" * 60)

    if bert_metrics:
        acc_gain = bert_metrics["accuracy"] - lexicon_metrics["accuracy"]
        f1_gain = bert_metrics["macro_f1"] - lexicon_metrics["macro_f1"]
        speedup = bert_time / lexicon_time if lexicon_time else float("inf")
        print(f"\nSummary:")
        print(f"  BERT accuracy gain : {acc_gain:+.4f}")
        print(f"  BERT macro F1 gain : {f1_gain:+.4f}")
        print(f"  BERT is {speedup:.1f}x slower than Lexicon")
        recommendation = "BERT" if f1_gain > 0.05 else "Lexicon (marginal gain not worth the compute cost)"
        print(f"  Recommendation     : {recommendation}")


if __name__ == "__main__":
    main()
