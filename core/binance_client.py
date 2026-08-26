"""
Binance Futures Data Client
Downloads historical OHLCV and funding rates from Binance (mainnet).
No API key required for public endpoints.
"""
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

BASE_URL = "https://fapi.binance.com"

class BinanceFuturesClient:
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{BASE_URL}{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def download_klines(
        self, 
        symbol: str, 
        interval: str = "1h", 
        months: int = 12
    ) -> pd.DataFrame:
        """Download historical OHLCV klines from Binance Futures."""
        logger.info(f"Downloading {symbol} {interval} klines for past {months} months...")

        # Binance kline limit per request = 1500
        # For 1h, 1500 bars = ~62 days
        # We'll paginate backward from now
        all_data = []
        end_time = None
        total_bars_target = months * 30 * 24  # approximate

        while len(all_data) < total_bars_target:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": 1500,
            }
            if end_time:
                params["endTime"] = end_time

            batch = self._get("/fapi/v1/klines", params)

            if not batch:
                break

            all_data.extend(batch)
            end_time = batch[0][0] - 1  # Move backward
            logger.debug(f"Fetched {len(batch)} bars. Total: {len(all_data)}")

            if len(batch) < 1500:
                break

        if not all_data:
            raise ValueError(f"No data returned for {symbol}")

        # Binance kline format:
        # [open_time, open, high, low, close, volume, close_time, quote_volume, 
        #  trades, taker_buy_base, taker_buy_quote, ignore]
        df = pd.DataFrame(all_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", 
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        # Types
        numeric_cols = ["open", "high", "low", "close", "volume", 
                       "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.set_index("open_time").sort_index()

        # Remove duplicates and NaNs
        df = df[~df.index.duplicated(keep="last")].dropna()

        logger.info(f"Downloaded {len(df)} rows for {symbol} ({df.index[0]} to {df.index[-1]})")
        return df

    def download_funding_rates(
        self, 
        symbol: str, 
        months: int = 12
    ) -> pd.DataFrame:
        """Download historical funding rates (every 8 hours)."""
        logger.info(f"Downloading {symbol} funding rates for past {months} months...")

        # Paginate forward from the start date. The public endpoint returns
        # records in ascending order when startTime is provided.
        start_time = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=months)
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

        all_data = []
        while start_ms < end_ms:
            params = {
                "symbol": symbol,
                "startTime": start_ms,
                "limit": 1000,
            }

            batch = self._get("/fapi/v1/fundingRate", params)

            if not batch:
                break

            all_data.extend(batch)
            start_ms = batch[-1]["fundingTime"] + 1
            logger.debug(f"Fetched {len(batch)} funding records. Total: {len(all_data)}")

            if len(batch) < 1000:
                break

        if not all_data:
            raise ValueError(f"No funding data for {symbol}")

        df = pd.DataFrame(all_data)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df = df.set_index("fundingTime").sort_index()
        df = df[~df.index.duplicated(keep="last")].dropna()

        logger.info(f"Downloaded {len(df)} funding records for {symbol} ({df.index[0]} to {df.index[-1]})")
        return df

    def save(self, df: pd.DataFrame, name: str):
        path = self.data_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved {name} to {path}")

    def load(self, name: str) -> Optional[pd.DataFrame]:
        path = self.data_dir / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None
