# Use a supported Debian release (buster is EOL)
FROM python:3.10-slim-bookworm

# Helpful defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install deps first for better layer caching
COPY requirements.txt /app/
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . /app

# Expose the port we’ll serve on
EXPOSE 8080

# If your FastAPI instance is named `app` in app/app.py:
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]

# If you actually run a top-level script instead, swap to:
# CMD ["python", "app.py"]
