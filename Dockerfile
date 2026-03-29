FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run inference and then keep alive with dummy server
CMD ["sh", "-c", "python inference.py; echo 'Evaluation Complete. Keeping Space Alive...'; python -m http.server 7860"]
