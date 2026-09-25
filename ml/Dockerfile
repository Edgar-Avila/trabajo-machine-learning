FROM python:3.11-slim
WORKDIR /ml
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY db.py features.py train.py seed.py predict.py ingest.py scheduler.py promote_model.py .