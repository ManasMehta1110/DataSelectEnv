FROM python:3.11.9-slim-bookworm

# HF Spaces runs as a non-root user — create it early
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HF Spaces expects port 7860
ENV PORT=7860

# Switch to non-root
USER appuser

EXPOSE 7860

CMD ["python", "server.py"]