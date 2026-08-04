FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py .

EXPOSE 7860
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
