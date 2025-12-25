# models/data_handler.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import requests_cache

class DataHandler:
    """Handles data downloading and processing."""

    @staticmethod
    def download_close(ticker, start, end):
        """Download stock price data and return Close prices."""
        try:
            session = requests_cache.CachedSession('yfinance.cache')
            session.headers['User-agent'] = 'Mozilla/5.0'

            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False, session=session)
            if df.empty:
                return None
            return df["Close"]
        except Exception as e:
            raise Exception(f"Error downloading data for {ticker}: {e}")
    
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