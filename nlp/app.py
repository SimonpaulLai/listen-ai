import os
import re
from collections import Counter

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="listen-ai-nlp-optimized")

# ── Lexicon-based classifier (original algorithm, kept as baseline) ──────────

POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "awesome", "happy",
    "amazing", "nice", "best", "positive", "fast", "smooth", "reliable",
}

POSITIVE_WORDS_ZH_TW = {
    "好", "很好", "優秀", "喜歡", "讚", "開心", "高興", "棒",
    "最佳", "正面", "快速", "順暢", "可靠", "滿意", "推薦",
    "優質", "超棒", "完美", "厲害", "超讚", "划算", "值得",
    "好用", "方便", "實用", "新鮮", "美味", "好吃", "愉快",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "slow", "bug", "bugs",
    "issue", "issues", "angry", "broken", "negative", "expensive",
}

NEGATIVE_WORDS_ZH_TW = {
    "差", "糟糕", "很糟", "討厭", "最差", "慢", "錯誤", "問題",
    "生氣", "壞掉", "負面", "昂貴", "失望", "卡頓", "爛", "崩潰",
    "無法", "不行", "惡劣", "浪費", "後悔", "詐騙", "假貨",
    "難吃", "臭", "噁心",
}

NEGATION_WORDS = {"not", "never", "no", "hardly", "不", "沒", "無", "未", "別", "不是"}

POSITIVE_WORDS_ALL = POSITIVE_WORDS | POSITIVE_WORDS_ZH_TW
NEGATIVE_WORDS_ALL = NEGATIVE_WORDS | NEGATIVE_WORDS_ZH_TW

CJK_LEXICON_TERMS = sorted(
    POSITIVE_WORDS_ZH_TW | NEGATIVE_WORDS_ZH_TW | {w for w in NEGATION_WORDS if re.search(r"[\u4e00-\u9fff]", w)},
    key=len, reverse=True,
)


def _tokenize_cjk_segment(segment: str) -> list[str]:
    tokens: list[str] = []
    idx = 0
    while idx < len(segment):
        match = ""
        for term in CJK_LEXICON_TERMS:
            if segment.startswith(term, idx):
                match = term
                break
        if match:
            tokens.append(match)
            idx += len(match)
        else:
            tokens.append(segment[idx])
            idx += 1
    return tokens


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower())
    tokens: list[str] = []
    for raw in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(_tokenize_cjk_segment(raw))
        else:
            tokens.append(raw)
    return tokens


def classify_text_lexicon(text: str) -> tuple[str, int]:
    tokens = tokenize(text)
    score = 0
    previous_tokens = ["", ""]
    for token in tokens:
        is_negated = any(prev in NEGATION_WORDS for prev in previous_tokens)
        if token in POSITIVE_WORDS_ALL:
            score += -1 if is_negated else 1
        elif token in NEGATIVE_WORDS_ALL:
            score += 1 if is_negated else -1
        previous_tokens = [previous_tokens[-1], token]
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", score


# ── BERT-based classifier (new algorithm) ────────────────────────────────────

BERT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
BERT_NEUTRAL_THRESHOLD = 0.60  # confidence below this → neutral

_bert_pipeline = None
USE_BERT = False

SENTIMENT_MODE = os.getenv("SENTIMENT_MODE", "bert").lower()

if SENTIMENT_MODE == "bert":
    try:
        from transformers import pipeline as hf_pipeline
        print(f"Loading BERT model: {BERT_MODEL} ...")
        _bert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model=BERT_MODEL,
            device=-1,  # CPU; set to 0 for CUDA
        )
        USE_BERT = True
        print("BERT model loaded successfully.")
    except Exception as exc:
        print(f"BERT load failed ({exc}), falling back to lexicon mode.")


def classify_text_bert(text: str) -> tuple[str, float]:
    assert _bert_pipeline is not None
    result = _bert_pipeline(text, truncation=True, max_length=512)[0]
    label_raw: str = result["label"].lower()   # "positive", "neutral", "negative"
    confidence: float = result["score"]
    if confidence < BERT_NEUTRAL_THRESHOLD:
        return "neutral", confidence
    return label_raw, confidence


# ── API models ────────────────────────────────────────────────────────────────

class SentimentRequest(BaseModel):
    texts: list[str]


class SentimentItem(BaseModel):
    text: str
    label: str
    score: float


class SentimentResponse(BaseModel):
    sentiment_percentage: dict[str, float]
    classifications: list[SentimentItem]
    model_used: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    mode = "bert" if USE_BERT else "lexicon"
    return {"status": "ok", "service": "nlp", "port": os.getenv("NLP_PORT", "8001"), "mode": mode}


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest) -> SentimentResponse:
    results: list[SentimentItem] = []
    counts = Counter({"positive": 0, "neutral": 0, "negative": 0})

    for text in req.texts:
        if USE_BERT:
            label, score = classify_text_bert(text)
        else:
            label, raw_score = classify_text_lexicon(text)
            score = float(raw_score)

        counts[label] += 1
        results.append(SentimentItem(text=text, label=label, score=score))

    total = max(1, len(req.texts))
    return SentimentResponse(
        sentiment_percentage={k: round((v / total) * 100, 2) for k, v in counts.items()},
        classifications=results,
        model_used="BERT" if USE_BERT else "Lexicon",
    )
