# nlp/app.py (更新後版本)
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from collections import Counter
from transformers import pipeline

app = FastAPI(title="listen-ai-nlp-optimized")

# --- 舊有的字典法邏輯 (保留作為對照組) ---
# ... (這裡放你原本的 POSITIVE_WORDS, tokenize, classify_text 等函式) ...

# --- 新的深度學習邏輯 (BERT) ---
# 使用針對中文情緒分析微調過的模型
MODEL_NAME = "shibing624/bert-base-chinese-sentiment"
try:
    print(f"Loading model {MODEL_NAME}...")
    # 使用 CPU 推理 (M2 16GB 跑這個很快)
    nlp_model = pipeline("sentiment-analysis", model=MODEL_NAME)
    USE_BERT = True
except Exception as e:
    print(f"Failed to load BERT: {e}")
    USE_BERT = False

class SentimentRequest(BaseModel):
    texts: list[str]

class SentimentItem(BaseModel):
    text: str
    label: str
    score: float # 深度學習改回傳信心分數

class SentimentResponse(BaseModel):
    sentiment_percentage: dict[str, float]
    classifications: list[SentimentItem]
    model_used: str

@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest) -> SentimentResponse:
    results: list[SentimentItem] = []
    counts = Counter({"positive": 0, "neutral": 0, "negative": 0})

    for text in req.texts:
        if USE_BERT:
            # BERT 預測
            prediction = nlp_model(text)[0]
            # 將模型的 LABEL_1 (正向) / LABEL_0 (負向) 轉回系統標籤
            label = "positive" if prediction['label'] == 'LABEL_1' else "negative"
            score = prediction['score']
        else:
            # 回退到字典法
            label, raw_score = classify_text(text)
            score = float(raw_score)

        counts[label] += 1
        results.append(SentimentItem(text=text, label=label, score=score))

    total = max(1, len(req.texts))
    return SentimentResponse(
        sentiment_percentage={k: round((v / total) * 100, 2) for k, v in counts.items()},
        classifications=results,
        model_used="BERT-DeepLearning" if USE_BERT else "Lexicon-Based"
    )