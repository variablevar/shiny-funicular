-- GT-Quant TimescaleDB multi-timeframe schema
-- Day 1: hypertables per granularity (1m -> 1d) plus derivatives tables.
-- Chunk intervals: 1m = 1 week, 5m = 1 month, higher TFs = 3 months.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------
-- OHLCV hypertables, one per timeframe.
-- 1m carries microstructure extension columns populated by the
-- Cryptofeed tick -> 1m bar builder (Day 2).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ohlcv_1m (
    time                TIMESTAMPTZ       NOT NULL,
    symbol              TEXT              NOT NULL,
    open                DOUBLE PRECISION,
    high                DOUBLE PRECISION,
    low                 DOUBLE PRECISION,
    close               DOUBLE PRECISION,
    volume              DOUBLE PRECISION,
    -- microstructure extensions (NULL for exchange-official backfill bars)
    spread              DOUBLE PRECISION,
    bid_ask_imbalance   DOUBLE PRECISION,
    volume_delta        DOUBLE PRECISION,
    tick_momentum       DOUBLE PRECISION,
    realized_variance   DOUBLE PRECISION,
    trade_count         INTEGER,
    UNIQUE (time, symbol)
);

CREATE TABLE IF NOT EXISTS ohlcv_5m  (time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, UNIQUE (time, symbol));
CREATE TABLE IF NOT EXISTS ohlcv_15m (time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, UNIQUE (time, symbol));
CREATE TABLE IF NOT EXISTS ohlcv_30m (time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, UNIQUE (time, symbol));
CREATE TABLE IF NOT EXISTS ohlcv_1h  (time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, UNIQUE (time, symbol));
CREATE TABLE IF NOT EXISTS ohlcv_4h  (time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, UNIQUE (time, symbol));
CREATE TABLE IF NOT EXISTS ohlcv_1d  (time TIMESTAMPTZ NOT NULL, symbol TEXT NOT NULL, open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION, volume DOUBLE PRECISION, UNIQUE (time, symbol));

SELECT create_hypertable('ohlcv_1m',  'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 week');
SELECT create_hypertable('ohlcv_5m',  'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');
SELECT create_hypertable('ohlcv_15m', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');
SELECT create_hypertable('ohlcv_30m', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '3 months');
SELECT create_hypertable('ohlcv_1h',  'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '3 months');
SELECT create_hypertable('ohlcv_4h',  'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '3 months');
SELECT create_hypertable('ohlcv_1d',  'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 year');

-- ---------------------------------------------------------------------
-- Derivatives tables
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS funding_rates (
    time          TIMESTAMPTZ NOT NULL,
    symbol        TEXT        NOT NULL,
    funding_rate  DOUBLE PRECISION,
    mark_price    DOUBLE PRECISION,
    UNIQUE (time, symbol)
);
SELECT create_hypertable('funding_rates', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '3 months');

CREATE TABLE IF NOT EXISTS open_interest (
    time           TIMESTAMPTZ NOT NULL,
    symbol         TEXT        NOT NULL,
    open_interest  DOUBLE PRECISION,
    UNIQUE (time, symbol)
);
SELECT create_hypertable('open_interest', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');

CREATE TABLE IF NOT EXISTS liquidations (
    time      TIMESTAMPTZ NOT NULL,
    symbol    TEXT        NOT NULL,
    side      TEXT,             -- BUY (short liq) / SELL (long liq)
    price     DOUBLE PRECISION,
    quantity  DOUBLE PRECISION
);
SELECT create_hypertable('liquidations', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 week');

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    bids        JSONB,
    asks        JSONB,
    sequence    BIGINT
);
SELECT create_hypertable('orderbook_snapshots', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- ---------------------------------------------------------------------
-- Audit log (Day 6): one row per decision with full TF context.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    time              TIMESTAMPTZ NOT NULL,
    pair              TEXT,
    timeframe         TEXT,
    signal_type       TEXT,
    model_prediction  DOUBLE PRECISION,
    regime            TEXT,
    spread_1m         DOUBLE PRECISION,
    volume_delta_1m   DOUBLE PRECISION,
    risk_decision     TEXT,
    fill_price        DOUBLE PRECISION,
    slippage          DOUBLE PRECISION,
    pnl               DOUBLE PRECISION,
    extra             JSONB
);
SELECT create_hypertable('audit_log', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');

-- Shadow log (Day 11): LLM vs quant decision comparison.
CREATE TABLE IF NOT EXISTS shadow_log (
    time              TIMESTAMPTZ NOT NULL,
    pair              TEXT,
    quant_signal      TEXT,
    llm_recommendation JSONB,
    agreement         BOOLEAN,
    llm_confidence    DOUBLE PRECISION
);
SELECT create_hypertable('shadow_log', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');

-- Indexes for the hot query paths.
CREATE INDEX IF NOT EXISTS idx_ohlcv_1m_symbol_time  ON ohlcv_1m  (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_symbol_time  ON ohlcv_5m  (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_funding_symbol_time   ON funding_rates (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_oi_symbol_time        ON open_interest (symbol, time DESC);
