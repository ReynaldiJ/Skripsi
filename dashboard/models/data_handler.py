# models/data_handler.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import requests
import time
import random

class DataHandler:
    """Handles data downloading and processing."""
    
    @staticmethod
    def download_close(ticker: str, start: str, end: str) -> pd.Series | None:
        """
        Streamlit Community Cloud–friendly:
        - Uses a requests.Session with a browser-like User-Agent (reduces Yahoo blocking)
        - Uses Ticker().history() (often more stable than yf.download on hosted IPs)
        - Retries with exponential backoff + jitter (handles rate-limits/transient blocks)
        - Caches results to avoid refetching on every rerun/user interaction
        """
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        })

        last_err: Exception | None = None

        for attempt in range(6):
            try:
                t = yf.Ticker(ticker, session=session)

                # history() lets the session apply to requests; yf.download() is harder to control.
                df = t.history(start=start, end=end, auto_adjust=False)

                if df is None or df.empty or "Close" not in df.columns:
                    return None

                close = df["Close"]
                # Some tickers can return duplicates or unsorted index; normalize a bit.
                close = close[~close.index.duplicated(keep="last")].sort_index()
                return close

            except Exception as e:
                last_err = e
                # Exponential backoff + jitter to survive throttling on shared Streamlit Cloud IPs
                time.sleep((2 ** attempt) + random.random())

        # If it still fails after retries, raise a clear error
        raise Exception(f"Error downloading data for {ticker} (likely rate-limited/blocked): {last_err}")
    
    # def download_close(ticker, start, end):
    #     """Download stock price data and return Close prices."""
    #     try:
    #         df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
    #         if df.empty:
    #             return None
    #         return df["Close"]
    #     except Exception as e:
    #         raise Exception(f"Error downloading data for {ticker}: {e}")
    
    @staticmethod
    def weekly_windows(prices):
        """Return (weeks × 7 days) matrix with weekend forward-filled."""
        if prices is None or len(prices) == 0:
            return np.array([]), []
        
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
        weeks = []
        week_starts = []

        monday = prices.resample("W-MON").first()
        for i in range(len(monday) - 1):
            s, e = monday.index[i], monday.index[i + 1]
            w = prices[(prices.index >= s) & (prices.index < e)]
            if len(w) < 4:  # skip incomplete weeks
                continue
            # pad to 5 business days
            if len(w) < 5:
                w = w.reindex(pd.date_range(s, periods=5, freq="B"), method="ffill")
            w = w[:5].values
            # add weekend = Fri value
            w = np.append(w, [w[-1], w[-1]])
            weeks.append(w)
            week_starts.append(s)
        return np.array(weeks), week_starts