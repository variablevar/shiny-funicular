# GT-Quant FreqAI image: stable_freqai + custom deps for the strategy layer.
# - psycopg2-binary: read 1m microstructure from TimescaleDB (entry filter)
FROM freqtradeorg/freqtrade:stable_freqai

USER root
RUN pip install --no-cache-dir psycopg2-binary
USER ftuser
