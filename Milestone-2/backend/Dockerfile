FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py vectorizer.joblib model.joblib ./

EXPOSE 8000
ENV WEB_CONCURRENCY=4
CMD ["sh", "-c", "gunicorn app:app -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY} -b 0.0.0.0:8000"]
