FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/^# *en_US.UTF-8 UTF-8/s/^# *//' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

COPY Milestone-2/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir skl2onnx scipy

COPY Milestone-2/retraining_monitor.py Milestone-2/retrain_cycle.py Milestone-2/label_with_llm.py ./
COPY Milestone-2/model_control.py Milestone-2/drift.py ./
COPY Milestone-2/drift_monitor.py Milestone-2/build_drift_reference.py ./
COPY Milestone-2/backend/database.py Milestone-2/backend/model_store.py ./
COPY Milestone-1/nlp-getting-started/train.csv ./train.csv

# Freeze the training distribution at build time. Drift is measured against the
# data the model was trained on, not against whatever ships in the image.
RUN python build_drift_reference.py \
    --training-data train.csv \
    --output artifacts/drift_reference.joblib

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

CMD ["python", "retraining_monitor.py"]