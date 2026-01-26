FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git postgresql-client && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN pip install poetry && poetry install --no-interaction --no-ansi
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
