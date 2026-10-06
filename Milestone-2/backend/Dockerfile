FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends locales \
	&& sed -i '/^# *en_US.UTF-8 UTF-8/s/^# *//' /etc/locale.gen \
	&& locale-gen en_US.UTF-8 \
	&& rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app.py backend/database.py backend/model_store.py ./
COPY model/model_fp32.onnx ./model/model_fp32.onnx

EXPOSE 8000
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV WEB_CONCURRENCY=4
CMD ["sh", "-c", "gunicorn app:app -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY} -b 0.0.0.0:8000"]