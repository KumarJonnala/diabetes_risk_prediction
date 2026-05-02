FROM python:3.11-slim

WORKDIR /app

# install dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source packages needed by api/app.py
COPY src/ ./src/
COPY configs/ ./configs/

# copy the API entrypoint
COPY api/app.py ./api/app.py

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
