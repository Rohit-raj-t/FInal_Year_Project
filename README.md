# FInal_Year_Project

FinSight — Realtime-first (Demo)
AI-Powered Stock Advisor (realtime-first)

A Streamlit demo that combines realtime price history, news analysis (via yfinance + Google News RSS fallback), a local cache for offline/demos, an LSTM time-series predictor, and an LLM-based news sentiment analyzer. The app prioritizes realtime data but gracefully falls back to RSS and local files when rate limits or network issues occur. It also visualizes the last 30 days of price data and produces a best-effort 30-day recursive forecast from the trained LSTM.

NOT financial advice. This is a research / demonstration project only.

Features
- Resolve free-text company names into tickers (US + Indian large-caps).
- Realtime-first data pipeline.
- LSTM time-series model for short-term forecasting.
- LLM-based news sentiment analysis.
- Retry/backoff for yfinance & LLM.
- Auto-save fetched history/news.
- Streamlit UI with 30-day real + 30-day forecast chart.

Repository Structure
.
├─ finsight_streamlit.py
├─ requirements.txt
├─ README.md
├─ data/
└─ news/

Requirements
streamlit
python-dotenv
pandas
numpy
requests
feedparser
beautifulsoup4
textblob
yfinance
matplotlib
torch
torchvision (optional)

Running
pip install -r requirements.txt
streamlit run finsight_streamlit.py

Environment Variables
LLM_HOST, LLM_ALT, LLM_MODEL_NEWS, AUTO_SAVE_FETCHED, DATA_DIR, NEWS_DIR, MAX_NEWS

Usage
Search ticker/company or scan best upcoming stock.
Shows:
- LSTM prediction
- News sentiment
- Combined signal
- Graph (last 30d + next 30d)
- JSON metadata

Local Caching
AUTO_SAVE_FETCHED=1 enables caching under ./data and ./news.

LLM Integration
Calls /api/generate with fallback heuristic.

Recommendation Logic
LSTM 30%, News 70%
Buy/Hold/Sell rules.

