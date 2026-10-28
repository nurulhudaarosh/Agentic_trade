"""Text annotation shared by every memory system: FinBERT sentiment + MiniLM embeddings.

All systems receive the same annotations, so performance differences come from the
memory architecture (what is stored, retired and retrieved), not from the encoder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FINBERT = "ProsusAI/finbert"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def item_text(headline: str, summary: str = "") -> str:
    text = headline.strip()
    if summary and summary.strip() and summary.strip() != text:
        text = f"{text}. {summary.strip()}"
    return text[:512]


def finbert_scores(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return an (n, 3) array of [positive, negative, neutral] probabilities."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    tokenizer = AutoTokenizer.from_pretrained(FINBERT)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    order = [labels.index("positive"), labels.index("negative"), labels.index("neutral")]
    out = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=96, return_tensors="pt")
            probs = torch.softmax(model(**batch).logits, dim=-1).numpy()
            out[start:start + len(probs)] = probs[:, order]
            if (start // batch_size) % 50 == 0:
                print(f"[finbert] {start + len(probs)}/{len(texts)}", flush=True)
    return out


def embed(texts: list[str], batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDER, device="cpu")
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return vectors.astype(np.float16)


def annotate_dataset(data_dir: Path) -> None:
    """Write ``news_annotated.parquet``, ``filings_annotated.parquet`` and ``*_emb.npy``."""
    news = pd.read_parquet(data_dir / "news.parquet")
    filings = pd.read_parquet(data_dir / "filings.parquet")
    news_out = data_dir / "news_annotated.parquet"
    if not news_out.exists():
        texts = [item_text(h, s) for h, s in zip(news["headline"], news["summary"])]
        probs = finbert_scores([item_text(h) for h in news["headline"]])
        news["p_pos"], news["p_neg"], news["p_neu"] = probs[:, 0], probs[:, 1], probs[:, 2]
        news["sentiment"] = news["p_pos"] - news["p_neg"]
        np.save(data_dir / "news_emb.npy", embed(texts))
        news.to_parquet(news_out, index=False)
    filings_out = data_dir / "filings_annotated.parquet"
    if not filings_out.exists():
        # Filing "sentiment" is the fundamental growth score (YoY revenue/income/EPS),
        # which is more faithful for numeric XBRL summaries than a headline classifier.
        filings["sentiment"] = filings["fundamental_score"].astype(float)
        np.save(data_dir / "filings_emb.npy", embed(filings["text"].tolist()))
        filings.to_parquet(filings_out, index=False)
