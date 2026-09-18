# GT-Quant FreqAI image: stable_freqai + custom deps for the strategy layer.
# - psycopg2-binary: read 1m microstructure from TimescaleDB (entry filter)
# - patches/: FreqAI 2026.7 prediction-pipeline fixes (commit 210928a).
#   Without these, live inference dies with "Pipeline expected Index(..., N)
#   but got Index(..., N+9)" because find_features() overwrites the trained
#   feature list with the live strategy feature set. See docs/freqai-setup-notes.md.
FROM freqtradeorg/freqtrade:stable_freqai

USER root
RUN pip install --no-cache-dir psycopg2-binary
COPY patches/ /tmp/patches/
RUN cp /tmp/patches/data_kitchen.py /freqtrade/freqtrade/freqai/data_kitchen.py \
    && cp /tmp/patches/BaseRegressionModel.py /freqtrade/freqtrade/freqai/base_models/BaseRegressionModel.py
USER ftuser
