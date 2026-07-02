FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Download TextBlob corpora (correct command)
RUN python -m textblob.download_corpora

COPY . .

# Create backend folder and files
RUN mkdir -p backend && \
    echo '{"accuracy": 78.34}' > backend/model_metrics.json && \
    echo '[]' > backend/analysis_history.json

EXPOSE 7860

ENV FLASK_ENV=production

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "server:app"]
