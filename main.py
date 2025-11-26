import os
import json
import re
import time
import math
import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import requests
import feedparser
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from textblob import TextBlob

import yfinance as yf
import yfinance.exceptions as yfe

import difflib

load_dotenv()

# ---------------------------
# Configuration (tweak via env)
# ---------------------------
LLM_HOST = os.getenv("LLM_HOST", "http://127.0.0.1:11434")
LLM_ALTERNATES = [os.getenv("LLM_ALT", "http://127.0.0.1:14088")]
LLM_MODEL_NEWS = os.getenv("LLM_MODEL_NEWS", "gemma3:4b")
MAX_NEWS = int(os.getenv("MAX_NEWS", "10"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
NEWS_DIR = Path(os.getenv("NEWS_DIR", "news"))
AUTO_SAVE_FETCHED = os.getenv("AUTO_SAVE_FETCHED", "0") in ("1", "true", "True")

RSS_SOURCE = os.getenv(
    "RSS_SOURCE",
    'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    txt = BeautifulSoup(str(text), "html.parser").get_text()
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def now_ts() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except Exception:
        return str(obj)

# ---------------------------
# Company name -> ticker mapping + helper
# ---------------------------
NAME_TO_TICKER = {
    "apple": "AAPL", "apple inc": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "microsoft corp": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvidia corp": "NVDA", "nvda": "NVDA",
    "alphabet": "GOOGL", "google": "GOOGL", "google inc": "GOOGL", "googl": "GOOGL", "goog": "GOOG",
    "amazon": "AMZN", "amazon.com": "AMZN", "amzn": "AMZN",
    "meta": "META", "facebook": "META", "facebook inc": "META", "meta platforms": "META",
    "tesla": "TSLA", "tesla inc": "TSLA", "tsla": "TSLA",
    "oracle": "ORCL", "oracle corp": "ORCL",
    "intel": "INTC", "intel corp": "INTC",
    "broadcom": "AVGO", "broadcom inc": "AVGO",
    "netflix": "NFLX", "netflix inc": "NFLX",
    "amd": "AMD", "advanced micro devices": "AMD",
    "salesforce": "CRM", "ibm": "IBM", "cisco": "CSCO",
    "exxon": "XOM", "chevron": "CVX",
    "walmart": "WMT", "disney": "DIS", "nike": "NKE",
    "jpmorgan": "JPM", "jpmorgan chase": "JPM", "jpm": "JPM",
    "morgan stanley": "MS", "bank of america": "BAC", "goldman sachs": "GS", "wells fargo": "WFC",
    "tcs": "TCS.NS", "tata consultancy services": "TCS.NS",
    "reliance": "RELIANCE.NS", "reliance industries": "RELIANCE.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS", "hdfc": "HDFC.NS",
    "icici bank": "ICICIBANK.NS", "icici": "ICICIBANK.NS",
    "larsen & toubro": "LT.NS", "lt": "LT.NS",
    "state bank of india": "SBIN.NS", "sbin": "SBIN.NS",
    "ongc": "ONGC.NS", "hindustan unilever": "HINDUNILVR.NS",
    "bharti airtel": "BHARTIARTL.NS", "bharti": "BHARTIARTL.NS",
    "coal india": "COALINDIA.NS", "adani enterprises": "ADANIENT.NS",
    "maruti": "MARUTI.NS", "maruti suzuki": "MARUTI.NS",
    "alphabet inc": "GOOGL", "alphabet inc.": "GOOGL",
    "apple inc.": "AAPL", "amazon inc": "AMZN", "microsoft corporation": "MSFT",
}

def _normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\.\,\'\"\(\)]", "", s)
    s = re.sub(r"\band\b", "&", s)
    s = re.sub(r"\s+", " ", s)
    return s

NORMALIZED_NAME_TO_TICKER = {_normalize_name(k): v for k, v in NAME_TO_TICKER.items()}

def company_name_to_ticker(name: str, prefer_market: Optional[str] = None) -> Optional[str]:
    """
    Resolve a user-supplied company name or ticker-like string to a ticker symbol.

    Important behavior change:
    - Prefer mapping/fuzzy match first.
    - Only accept literal ticker if it already looks like one (uppercase or contains a dot).
    """
    if not name:
        return None

    name_stripped = name.strip()

    n = _normalize_name(name)
    if n in NORMALIZED_NAME_TO_TICKER:
        cand = NORMALIZED_NAME_TO_TICKER[n]
        if prefer_market and prefer_market.upper() == "IN":
            base = cand.split(".")[0]
            alt = f"{base}.NS"
            if alt in NORMALIZED_NAME_TO_TICKER.values():
                return alt
        return cand

    tokens = [t for t in re.split(r"\s+|[,/&-]", n) if t]
    for key, tk in NORMALIZED_NAME_TO_TICKER.items():
        if all(tok in key for tok in tokens):
            return tk

    candidates = difflib.get_close_matches(n, NORMALIZED_NAME_TO_TICKER.keys(), n=3, cutoff=0.7)
    if candidates:
        return NORMALIZED_NAME_TO_TICKER[candidates[0]]

    klein = re.sub(r"\b(inc|incorporated|ltd|limited|corp|corporation|co|plc|private)\b", "", n).strip()
    if klein and klein != n:
        if klein in NORMALIZED_NAME_TO_TICKER:
            return NORMALIZED_NAME_TO_TICKER[klein]
        candidates = difflib.get_close_matches(klein, NORMALIZED_NAME_TO_TICKER.keys(), n=2, cutoff=0.7)
        if candidates:
            return NORMALIZED_NAME_TO_TICKER[candidates[0]]

    if re.fullmatch(r"[A-Za-z0-9\-]{1,8}(\.[A-Za-z]{1,3})?", name_stripped):
        looks_upper = (name_stripped == name_stripped.upper())
        has_dot = "." in name_stripped
        if looks_upper or has_dot:
            return name_stripped.upper()

    return None

# ---------------------------
# Local-file loaders & savers
# ---------------------------
def load_local_history(ticker: str) -> Optional[pd.DataFrame]:
    candidates = [
        DATA_DIR / f"{ticker.upper()}.csv",
        DATA_DIR / f"{ticker.upper()}_history.csv",
        DATA_DIR / f"{ticker}.csv",
        DATA_DIR / f"{ticker}_history.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                df.columns = [c.strip() for c in df.columns]
                if "Close" not in df.columns and "Adj Close" in df.columns:
                    df["Close"] = df["Adj Close"]
                if "close" in df.columns and "Close" not in df.columns:
                    df.rename(columns={"close": "Close"}, inplace=True)
                if "volume" in df.columns and "Volume" not in df.columns:
                    df.rename(columns={"volume": "Volume"}, inplace=True)
                if "Close" in df.columns and "Volume" in df.columns:
                    return df.sort_index()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    if "Close" not in df.columns:
                        df["Close"] = df[numeric_cols[-1]]
                    if "Volume" not in df.columns and len(numeric_cols) > 1:
                        df["Volume"] = df[numeric_cols[0]]
                    if "Close" in df.columns and "Volume" in df.columns:
                        return df[["Close", "Volume"]].sort_index()
            except Exception as e:
                st.warning(f"Failed to read local history {p.name}: {e}")
    return None

def save_local_history(ticker: str, df: pd.DataFrame):
    if not AUTO_SAVE_FETCHED:
        return
    try:
        p = DATA_DIR / f"{ticker.upper()}.csv"
        df.to_csv(p)
    except Exception as e:
        st.warning(f"Failed to save history for {ticker}: {e}")

def load_local_news(ticker: str) -> List[str]:
    candidates = [
        NEWS_DIR / f"{ticker.upper()}_news.json",
        NEWS_DIR / f"{ticker.upper()}.json",
        NEWS_DIR / f"{ticker}_news.json",
        NEWS_DIR / f"{ticker}.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                headlines = raw.get("headlines") or raw.get("items") or raw.get("news") or raw.get("data") or []
                if isinstance(headlines, list) and headlines:
                    return [clean_text(h) for h in headlines if h]
            except Exception as e:
                st.warning(f"Failed to read local news {p.name}: {e}")
    return []

def save_local_news(ticker: str, headlines: List[str]):
    if not AUTO_SAVE_FETCHED:
        return
    try:
        p = NEWS_DIR / f"{ticker.upper()}_news.json"
        p.write_text(json.dumps({"headlines": headlines}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"Failed to save news for {ticker}: {e}")

# ---------------------------
# Realtime fetch helpers (preferred)
# ---------------------------
def _fetch_with_retry(fn, retries:int=3, backoff:float=1.5, min_sleep:float=0.8):
    attempt = 0
    while attempt <= retries:
        try:
            return fn(), None
        except yfe.YFRateLimitError as e:
            attempt += 1
            if attempt > retries:
                return None, f"YFRateLimitError: {e}"
            time.sleep(min_sleep * (backoff ** (attempt - 1)))
        except requests.RequestException as e:
            attempt += 1
            if attempt > retries:
                return None, f"NetworkError: {e}"
            time.sleep(min_sleep * (backoff ** (attempt - 1)))
        except Exception as e:
            return None, f"OtherError: {e}"
    return None, "UnknownError"

@st.cache_data(show_spinner=False)
def fetch_history_realtime(ticker: str, period: str = "1y", auto_adjust: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    t = yf.Ticker(ticker)
    df, err = _fetch_with_retry(lambda: t.history(period=period, auto_adjust=auto_adjust), retries=3)
    if err:
        return None, err
    try:
        save_local_history(ticker, df)
    except Exception:
        pass
    return df, None

@st.cache_data(show_spinner=False)
def fetch_news_realtime(ticker: str, max_news: int = MAX_NEWS) -> Tuple[List[str], Optional[str]]:
    t = yf.Ticker(ticker)
    items, err = _fetch_with_retry(lambda: t.get_news() if hasattr(t, "get_news") else getattr(t, "news", []), retries=2)
    if err:
        return [], err
    headlines: List[str] = []
    for it in (items or [])[:max_news]:
        title = ""
        if isinstance(it, dict):
            title = it.get("title") or it.get("headline") or it.get("summary") or ""
        else:
            title = str(it)
        title = clean_text(title)
        if title and title not in headlines:
            headlines.append(title)
    if headlines:
        save_local_news(ticker, headlines)
    return headlines, None

# ---------------------------
# RSS fallback for news
# ---------------------------
def fetch_google_rss(ticker_or_query: str, max_news: int = MAX_NEWS) -> Tuple[List[str], Optional[str]]:
    try:
        query = f"{ticker_or_query} stock earnings OR revenue OR partnership OR acquisition OR lawsuit"
        url = RSS_SOURCE.format(query=requests.utils.requote_uri(query))
        feed = feedparser.parse(url)
        headlines: List[str] = []
        for entry in feed.entries[:max_news]:
            title = entry.get("title") or entry.get("summary") or ""
            title = clean_text(title)
            if title and title not in headlines:
                headlines.append(title)
        if headlines:
            save_local_news(ticker_or_query, headlines)
        return headlines, None
    except Exception as e:
        return [], f"RSSError: {e}"

# ---------------------------
# LLM & fallback heuristic (unchanged)
# ---------------------------
def _try_llm_request(host: str, model: str, prompt: str, timeout: int = 60, retries:int=2, backoff:float=2.0):
    attempt = 0
    while attempt <= retries:
        try:
            resp = requests.post(
                f"{host.rstrip('/')}/api/generate",
                json={"model": model, "prompt": prompt, "format": "json", "stream": False},
                timeout=timeout
            )
            if resp.status_code >= 500:
                attempt += 1
                if attempt > retries:
                    return None, f"LLMServerError {resp.status_code}: {resp.text[:200]}"
                time.sleep((backoff ** attempt))
                continue
            return resp, None
        except requests.ReadTimeout as e:
            attempt += 1
            if attempt > retries:
                return None, f"LLMTimeout: {e}"
            time.sleep((backoff ** attempt))
        except requests.RequestException as e:
            attempt += 1
            if attempt > retries:
                return None, f"LLMRequestError: {e}"
            time.sleep((backoff ** attempt))
        except Exception as e:
            return None, f"LLMOtherError: {e}"
    return None, "LLMUnknownError"

def _find_first_json_like(text: str) -> Optional[str]:
    start = None
    level = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            level += 1
        elif ch == "}":
            level -= 1
            if level == 0 and start is not None:
                candidate = text[start:i+1]
                if '"' in candidate:
                    return candidate
                start = None
    return None

def _fallback_news_analysis(headlines: List[str]) -> Dict[str, Any]:
    if not headlines:
        return {"overall_sentiment": 0.0, "direction_probability": 0.5, "top_events": [], "summary": "No headlines found.", "confidence": 0.2}
    tb_scores = [TextBlob(h).sentiment.polarity for h in headlines]
    tb_mean = float(sum(tb_scores) / len(tb_scores))
    keywords_pos = ["launch", "acquire", "profit", "record", "beat", "growth", "surge", "increase", "partnership"]
    keywords_neg = ["lawsuit", "fine", "layoff", "miss", "down", "decline", "loss", "drop", "cut", "recall"]
    pos = sum(any(k in h.lower() for k in keywords_pos) for h in headlines)
    neg = sum(any(k in h.lower() for k in keywords_neg) for h in headlines)
    total = max(pos + neg, 1)
    heuristic = (pos - neg) / total
    overall = (tb_mean + heuristic) / 2.0
    direction_prob = (overall + 1.0) / 2.0
    top_events = [{"headline": h} for h in headlines[:3]]
    return {"overall_sentiment": float(max(-1.0, min(1.0, overall))), "direction_probability": float(max(0.0, min(1.0, direction_prob))), "top_events": top_events, "summary": "Fallback summary based on polarity and keywords.", "confidence": 0.3}

def analyze_with_llm(ticker: str, headlines: List[str]) -> Dict[str, Any]:
    if not headlines:
        return _fallback_news_analysis([])
    prompt = f"""
You are a financial analyst AI.
Analyze how the following recent news headlines affect {ticker}'s stock price.

Respond with EXACTLY ONE valid JSON object (no surrounding text or markdown) with these fields:
- overall_sentiment: float (range -1 to 1)
- direction_probability: float (range 0 to 1)
- top_events: list of {{ "headline": str, "impact_reason": str, "impact_direction": "positive"|"negative" }}
- summary: str
- confidence: float (0 to 1)

Headlines:
{json.dumps(headlines, ensure_ascii=False)}
"""
    hosts_to_try = [LLM_HOST] + [h for h in LLM_ALTERNATES if h and h != LLM_HOST]
    last_err = None
    for host in hosts_to_try:
        resp, err = _try_llm_request(host, LLM_MODEL_NEWS, prompt, timeout=60, retries=2)
        if err:
            last_err = f"{host}: {err}"
            continue
        raw_text = (resp.text or "").strip()
        try:
            parsed_top = resp.json()
            if isinstance(parsed_top, dict):
                if "response" in parsed_top:
                    inner = parsed_top["response"]
                    if isinstance(inner, dict):
                        return inner
                    if isinstance(inner, str):
                        try:
                            return json.loads(inner)
                        except Exception:
                            pass
                if any(k in parsed_top for k in ("overall_sentiment", "direction_probability", "summary")):
                    return parsed_top
        except Exception:
            pass
        candidate = _find_first_json_like(raw_text)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                pass
        last_err = f"{host}: Unable to parse LLM response"
    st.warning(f"LLM failed ({last_err}) — using fallback heuristic.")
    return _fallback_news_analysis(headlines)

# ---------------------------
# LSTM model & pipeline (realtime-first, local fallback)
# ---------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class SimpleLSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int=32, num_layers:int=1, out_size:int=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pct_change"] = df["Close"].pct_change().fillna(0.0)
    df["sma10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["sma20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-8)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    df["vol_change"] = df["Volume"].pct_change().fillna(0.0)
    df.fillna(method="bfill", inplace=True)
    df.fillna(0.0, inplace=True)
    return df

def build_sequences(df: pd.DataFrame, feature_cols: List[str], window:int=60, horizon:int=7) -> Tuple[np.ndarray, np.ndarray]:
    arr = df[feature_cols].values
    close = df["Close"].values
    sequences, targets = [], []
    N = len(df)
    for i in range(N - window - horizon + 1):
        seq = arr[i:i+window]
        future_close = close[i + window + horizon - 1]
        current_close = close[i + window - 1]
        pct_change = (future_close - current_close) / (current_close + 1e-8)
        sequences.append(seq)
        targets.append(pct_change)
    if not sequences:
        return np.empty((0, window, len(feature_cols))), np.array([])
    return np.array(sequences), np.array(targets)

@st.cache_resource(show_spinner=False)
def train_lstm_for_ticker(ticker: str, window:int=60, horizon:int=7, epochs:int=8, lr:float=1e-3, hidden_size:int=32) -> Tuple[SimpleLSTM, Dict[str,Any]]:
    df, err = fetch_history_realtime(ticker, period="5y")
    if err or df is None or df.empty or len(df) < (window + horizon + 10):
        df1, err1 = fetch_history_realtime(ticker, period="1y")
        if df1 is not None and not df1.empty:
            df = df1
            err = err1
        else:
            df_local = load_local_history(ticker)
            if df_local is not None:
                df = df_local.copy()
                err = None
            else:
                raise RuntimeError(f"No sufficient history (realtime error: {err} {err1 if 'err1' in locals() else ''}) and no local CSV found for {ticker}")

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Volume" not in df.columns:
        df["Volume"] = df.get("Volume", 0.0).fillna(0.0)

    df = df[["Close", "Volume"]].copy()
    df = compute_technical_indicators(df)
    feature_cols = ["Close", "Volume", "pct_change", "sma10", "sma20", "rsi14", "vol_change"]

    seqs, targets = build_sequences(df, feature_cols, window=window, horizon=horizon)
    if seqs.shape[0] < 10:
        window = max(10, min(window, max(10, int(len(df)/4))))
        seqs, targets = build_sequences(df, feature_cols, window=window, horizon=horizon)
    if seqs.shape[0] == 0:
        raise RuntimeError("Not enough sequence data for training")

    mean = seqs.mean(axis=(0,1), keepdims=True)
    std = seqs.std(axis=(0,1), keepdims=True) + 1e-8
    seqs_norm = (seqs - mean) / std

    N = len(seqs_norm)
    n_train = max(1, int(N * 0.7))
    n_val = max(0, int(N * 0.15))
    train_x, train_y = seqs_norm[:n_train], targets[:n_train]
    val_x, val_y = seqs_norm[n_train:n_train+n_val], targets[n_train:n_train+n_val]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size=seqs_norm.shape[2], hidden_size=hidden_size)
    model.to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TimeSeriesDataset(train_x, train_y)
    val_ds = TimeSeriesDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True) if len(train_ds) > 0 else []
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False) if len(val_ds) > 0 else []

    best_val_loss = float("inf")
    best_state = None
    for ep in range(max(1, epochs)):
        model.train()
        total_loss = 0.0
        if train_loader:
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += float(loss.item()) * xb.size(0)
        avg_train = total_loss / max(1, len(train_loader.dataset)) if train_loader else 0.0
        model.eval()
        vloss = 0.0
        if val_loader:
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device).unsqueeze(1)
                    pred = model(xb)
                    vloss += float(loss_fn(pred, yb).item()) * xb.size(0)
            avg_val = vloss / max(1, len(val_loader.dataset))
        else:
            avg_val = avg_train
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {"feature_cols": feature_cols, "mean": mean.tolist(), "std": std.tolist(), "window": window, "horizon": horizon, "trained_on": len(df)}
    return model, meta

def predict_with_lstm(model: SimpleLSTM, meta: Dict[str,Any], ticker: str) -> Dict[str,Any]:
    window = int(meta["window"]); feature_cols = meta["feature_cols"]
    df, err = fetch_history_realtime(ticker, period=f"{max(365, window*4)}d")
    if err or df is None or df.empty or len(df) < window:
        df_local = load_local_history(ticker)
        if df_local is not None:
            df = df_local.copy()
        else:
            raise RuntimeError(f"Not enough history to predict (realtime err: {err}) and no local CSV found")
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Volume" not in df.columns:
        df["Volume"] = df.get("Volume", 0.0).fillna(0.0)
    df = df[["Close", "Volume"]].copy()
    df = compute_technical_indicators(df)
    seq = df[feature_cols].values[-window:, :]
    mean = np.array(meta["mean"]); std = np.array(meta["std"]) + 1e-8
    try:
        mean_arr = np.squeeze(mean, axis=0)
    except Exception:
        mean_arr = mean
    try:
        std_arr = np.squeeze(std, axis=0)
    except Exception:
        std_arr = std
    seq_norm = (seq - mean_arr) / std_arr
    seq_norm = seq_norm.astype(np.float32)[None, :, :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(seq_norm).to(device)
        pred = model(xb).cpu().numpy().squeeze()
    predicted_change = float(pred * 100.0)
    prob_up = 1.0 / (1.0 + math.exp(-predicted_change / 5.0))
    return {"predicted_change": predicted_change, "prob_up": prob_up, "raw": float(pred)}

# ---------------------------
# Amalgamation & scoring, candidate scanning
# ---------------------------
LSTM_WEIGHT = 0.3; NEWS_WEIGHT = 0.7; BUY_THRESHOLD = 0.05; SELL_THRESHOLD = -0.05

def normalize_lstm_score(lstm_out: Dict[str,Any]) -> Dict[str,Any]:
    if not lstm_out:
        return {"score":0.0,"predicted_change_pct":None,"confidence":0.25}
    if "prob_up" in lstm_out and lstm_out["prob_up"] is not None:
        prob = max(0.0, min(1.0, float(lstm_out["prob_up"])))
        score = prob * 2 - 1
        predicted_change_pct = lstm_out.get("predicted_change")
        return {"score":score,"predicted_change_pct":predicted_change_pct,"confidence":0.6}
    if "predicted_change" in lstm_out and lstm_out["predicted_change"] is not None:
        pc = float(lstm_out["predicted_change"]); score = max(-1.0,min(1.0,pc/20.0))
        return {"score":score,"predicted_change_pct":pc,"confidence":0.6}
    return {"score":0.0,"predicted_change_pct":None,"confidence":0.25}

def normalize_news_score(news_out: Dict[str,Any]) -> Dict[str,Any]:
    if not news_out:
        return {"score":0.0,"confidence":0.25,"summary":None,"top_events":None}
    if news_out.get("overall_sentiment") is not None:
        s = max(-1.0,min(1.0,float(news_out["overall_sentiment"])))
        conf = float(news_out.get("confidence",0.6))
        return {"score":s,"confidence":conf,"summary":news_out.get("summary"),"top_events":news_out.get("top_events")}
    if news_out.get("direction_probability") is not None:
        p = max(0.0,min(1.0,float(news_out["direction_probability"]))); s = p*2-1
        conf = float(news_out.get("confidence",0.5))
        return {"score":s,"confidence":conf,"summary":news_out.get("summary"),"top_events":news_out.get("top_events")}
    return {"score":0.0,"confidence":0.25,"summary":news_out.get("summary"),"top_events":news_out.get("top_events")}

def compute_signal(lstm_score: float, news_score: float, w_lstm: float = LSTM_WEIGHT, w_news: float = NEWS_WEIGHT) -> float:
    raw = w_lstm * lstm_score + w_news * news_score
    return max(-1.0, min(1.0, raw))

def map_signal(signal: float) -> str:
    if signal > BUY_THRESHOLD: return "Buy"
    if signal < SELL_THRESHOLD: return "Sell"
    return "Hold"

def amalgamate(original_query: str, lstm_out: Dict[str,Any], news_out: Dict[str,Any]) -> Dict[str,Any]:
    lstm_meta = normalize_lstm_score(lstm_out)
    news_meta = normalize_news_score(news_out)
    signal = compute_signal(lstm_meta["score"], news_meta["score"])
    recommendation = map_signal(signal)
    combined_conf = (LSTM_WEIGHT * lstm_meta["confidence"]) + (NEWS_WEIGHT * news_meta["confidence"])
    predicted_change_pct = lstm_meta.get("predicted_change_pct")
    if predicted_change_pct is None:
        predicted_change_pct = round(signal * 5.0, 3)
    explanation = (f"Combined signal {signal:+.3f} → {recommendation}. "
                   f"LSTM score {lstm_meta['score']:+.3f}, News score {news_meta['score']:+.3f}. "
                   f"Predicted change ≈ {predicted_change_pct:+.2f}%")
    return {"timestamp": now_ts(), "original_query":original_query, "signal":round(signal,4),
            "recommendation":recommendation, "predicted_change_pct":predicted_change_pct,
            "confidence": round(max(0.0,min(1.0,combined_conf)),3), "explanation":explanation,
            "lstm_meta":lstm_meta, "news_meta":news_meta}

BIG_COMPANIES = ["AAPL","MSFT","NVDA","GOOG","AMZN","META","TSLA","ORCL","INTC","AVGO"]

@st.cache_data(show_spinner=False)
def evaluate_candidates_by_news(candidates: List[str], top_k:int=3) -> List[Dict[str,Any]]:
    results = []
    for idx, t in enumerate(candidates):
        time.sleep(0.15 + (idx * 0.02))
        headlines, err = fetch_news_realtime(t, max_news=MAX_NEWS)
        if err or not headlines:
            rss_headlines, rss_err = fetch_google_rss(t, max_news=MAX_NEWS)
            if rss_headlines:
                headlines = rss_headlines
                err = None
            else:
                local = load_local_news(t)
                if local:
                    headlines = local
                    err = None
        if err:
            results.append({"ticker": t, "news_score": -1.0, "error": err})
            continue
        news_out = analyze_with_llm(t, headlines)
        news_meta = normalize_news_score(news_out)
        info = {}
        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}
        results.append({"ticker":t, "news_score":news_meta["score"], "news_confidence":news_meta["confidence"], "summary":news_meta.get("summary"), "top_events":news_meta.get("top_events"), "market_cap": info.get("marketCap", None)})
    results_sorted = sorted(results, key=lambda x: x.get("news_score", -1.0), reverse=True)
    return results_sorted[:top_k]

# ---------------------------
# Streamlit UI (RESOLVE company names -> ticker BEFORE doing work)
# ---------------------------
st.set_page_config(page_title="FinSight — Realtime-first", layout="wide")
st.title("FinSight — An Multi Source AI Engine For Stock Trend Forcasting")

st.markdown(f"""
This demo **prefers realtime data from yfinance**. If realtime fetches fail (rate limits / network),
it falls back to **Google News RSS** for headlines and then to **local files** under `{DATA_DIR}` and `{NEWS_DIR}` for history/news.
""")
st.header("Query & Prediction")
mode = st.radio("Mode:", ["Search ticker", "Best upcoming stock (big companies)"])

if mode == "Search ticker":
    user_input = st.text_input("Enter ticker or company name (e.g. AAPL or Google)", value="AAPL")
    horizon_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=7)
    run_button = st.button("Run pipeline (realtime preferred)")

    if run_button and user_input:
        resolved = company_name_to_ticker(user_input)
        used_ticker = None
        if resolved:
            used_ticker = resolved
            st.info(f"Interpreted input '{user_input}' as ticker: {used_ticker}")
        else:
            cand = user_input.strip()
            # only accept as literal ticker if already uppercase or contains market dot
            if re.fullmatch(r"[A-Za-z0-9\-]{1,8}(\.[A-Za-z]{1,3})?", cand) and (cand == cand.upper() or "." in cand):
                used_ticker = cand.upper()
                st.warning(f"Couldn't map company name '{user_input}' to a known ticker. Proceeding with '{used_ticker}' (may fail).")
            else:
                suggestions = difflib.get_close_matches(_normalize_name(user_input), list(NORMALIZED_NAME_TO_TICKER.keys()), n=3, cutoff=0.6)
                if suggestions:
                    suggestion_ticks = [NORMALIZED_NAME_TO_TICKER[s] for s in suggestions]
                    st.warning(f"Couldn't confidently map '{user_input}'. Did you mean: {', '.join(suggestion_ticks)} ? Proceeding with your original input uppercased.")
                else:
                    st.warning(f"Couldn't map '{user_input}' to a ticker. Proceeding with your input uppercased (may fail).")
                used_ticker = user_input.strip().upper()

        ticker_input = used_ticker

        with st.spinner("Running pipeline (train LSTM on realtime history preferred, fetch news realtime preferred)..."):
            model, meta = None, None
            try:
                model, meta = train_lstm_for_ticker(ticker_input, window=60, horizon=horizon_days, epochs=6)
            except Exception as e:
                st.error(f"Error training/loading LSTM for {ticker_input}: {e}")
                model, meta = None, None

            lstm_out = {}
            if model is not None and meta is not None:
                try:
                    lstm_out = predict_with_lstm(model, meta, ticker_input)
                except Exception as e:
                    st.warning(f"LSTM predict failed: {e}")
                    lstm_out = {}

            headlines, news_err = fetch_news_realtime(ticker_input, max_news=MAX_NEWS)
            if news_err or not headlines:
                rss_headlines, rss_err = fetch_google_rss(ticker_input, max_news=MAX_NEWS)
                if rss_headlines:
                    headlines = rss_headlines
                    news_err = None
                else:
                    local = load_local_news(ticker_input)
                    if local:
                        headlines = local
                        news_err = None

            if news_err:
                st.warning(f"News fetch encountered: {news_err}")

            st.subheader("Recent headlines (realtime preferred, RSS/local fallback)")
            if headlines:
                for h in headlines:
                    st.write("•", h)
            else:
                st.write("No headlines found (realtime/RSS/local all failed or empty).")

            news_out = analyze_with_llm(ticker_input, headlines)
            amalgam = amalgamate(f"Prediction for {ticker_input}", lstm_out, news_out)

            # ---------------------------
            # Visualization: Past 30 days + Next 30 predicted days
            # ---------------------------
            st.subheader("📈 Price Chart: Last 30 Days + Next 30 Days Prediction")
            try:
                # Fetch recent history (90 days to be safe)
                hist_df, err_hist = fetch_history_realtime(ticker_input, period="90d")
                if err_hist or hist_df is None or hist_df.empty:
                    hist_df = load_local_history(ticker_input)

                if hist_df is not None and not hist_df.empty:
                    # normalize close column
                    if "Close" not in hist_df.columns and "Adj Close" in hist_df.columns:
                        hist_df["Close"] = hist_df["Adj Close"]
                    # ensure Volume column exists
                    if "Volume" not in hist_df.columns:
                        hist_df["Volume"] = hist_df.get("Volume", 0.0).fillna(0.0)

                    # prepare past series
                    past_df = hist_df.sort_index()
                    past_30 = past_df["Close"].tail(30)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(past_30.index, past_30.values, label="Past 30 Days", linewidth=2)

                    if model is not None and meta is not None:
                        window = int(meta.get("window", 60))
                        feature_cols = meta.get("feature_cols", ["Close", "Volume", "pct_change", "sma10", "sma20", "rsi14", "vol_change"])
                        df_for_feat = compute_technical_indicators(hist_df.copy())
                        if len(df_for_feat) >= window:
                            recent_window = df_for_feat[feature_cols].values[-window:, :].astype(np.float32)
                            mean = np.squeeze(np.array(meta["mean"]), axis=0)
                            std = np.squeeze(np.array(meta["std"]), axis=0) + 1e-8
                            seq_buffer = (recent_window - mean) / std  # normalized shape (window, features)

                            future_points = 30
                            future_prices = []
                            current_close = hist_df["Close"].iloc[-1]
                            last_vol = float(hist_df["Volume"].iloc[-1]) if "Volume" in hist_df.columns else 0.0

                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model.to(device)
                            model.eval()
                            for _ in range(future_points):
                                xb = torch.from_numpy(seq_buffer[None, :, :]).float().to(device)
                                with torch.no_grad():
                                    pred = model(xb).cpu().numpy().squeeze()
                                try:
                                    next_price = float(current_close) * (1.0 + float(pred))
                                except Exception:
                                    next_price = float(current_close) * (1.0 + float(pred))
                                future_prices.append(next_price)
                                pct_change = float(pred)
                                sma10 = next_price
                                sma20 = next_price
                                rsi14 = 50.0
                                vol_change = 0.0
                                new_row = np.array([next_price, last_vol, pct_change, sma10, sma20, rsi14, vol_change], dtype=np.float32)

                                new_row_norm = (new_row - mean) / std
                                seq_buffer = np.vstack([seq_buffer[1:], new_row_norm])
                                current_close = next_price

                            future_index = pd.date_range(start=past_30.index[-1] + pd.Timedelta(days=1), periods=len(future_prices), freq="D")
                            ax.plot(future_index, future_prices, linestyle="--", linewidth=2, label="Next 30 Days Prediction")
                        else:
                            st.info(f"Not enough history to produce multi-step forecast (need window={window} rows). Showing only past 30 days.")
                    else:
                        st.info("Model not available — showing only past 30 days.")

                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.set_title(f"{ticker_input} - Past 30 Days & Next 30 Days Forecast")
                    ax.legend()
                    fig.autofmt_xdate()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Cannot generate the graph — no historical data available (realtime & local fallback failed).")
            except Exception as e:
                st.warning(f"Graph generation failed: {e}")

            # ---------------------------
            # Result display
            # ---------------------------
            st.subheader("Result")
            st.metric("Recommendation", amalgam["recommendation"])
            st.write("Signal:", amalgam["signal"], "Confidence:", amalgam["confidence"])
            st.write("Predicted change (%) ≈", amalgam["predicted_change_pct"])
            st.markdown("**Explanation:**")
            st.write(amalgam["explanation"])

            st.subheader("Detailed metadata (JSON)")
            st.text("LSTM output (json):")
            st.code(safe_json_dumps(lstm_out), language="json")
            st.text("News analysis (json):")
            st.code(safe_json_dumps(news_out), language="json")

else:
    st.header("Scan big companies for promising news (realtime-first)")
    top_k = st.slider("How many top picks to show", min_value=1, max_value=5, value=3)
    scan_button = st.button("Find best upcoming stock (realtime preferred)")

    if scan_button:
        with st.spinner("Evaluating candidate companies (realtime news preferred)..."):
            picks = evaluate_candidates_by_news(BIG_COMPANIES, top_k=top_k)
        st.subheader("Top picks (by news sentiment score)")
        for p in picks:
            st.markdown(f"### {p.get('ticker')}")
            st.write(f"News score: {p.get('news_score'):+.3f}   |   Confidence: {p.get('news_confidence')}")
            if p.get("summary"):
                st.write("Summary:", p.get("summary"))
            if p.get("top_events"):
                st.write("Top events:", p.get("top_events"))
            mc = p.get("market_cap")
            if mc:
                st.write("Market cap:", f"{mc:,}")
            if p.get("error"):
                st.warning(f"Error for {p.get('ticker')}: {p.get('error')}")
            st.write("---")


st.markdown("---")
st.caption("FinSight — realtime-first demo. Built for demonstration and discussion; not production-ready.")
