# Multi-stage Dockerfile for FaceCV
# Stage 1: Base dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libdc1394-25 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as dependencies

WORKDIR /app

# Copy requirements files
COPY requirements-docker.txt ./requirements.txt

# Install Python dependencies with timeout and retry
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout 600 -r requirements.txt

# Stage 3: Application
FROM dependencies as app

WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=7000

# Create non-root user
RUN useradd -m -u 1000 facecv && \
    chown -R facecv:facecv /app

USER facecv

# Expose port
EXPOSE 7000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7000/health').raise_for_status()"

# Default command
CMD ["python", "main.py"]

# Stage 4: Development image with additional tools
FROM app as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dev dependencies
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

USER facecv

# Development command with auto-reload
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--reload"]

# Stage 5: Production optimized image
FROM app as production

# Copy only necessary files (exclude tests, docs, etc.)
USER root
RUN find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf tests/ docs/ *.md .git* && \
    chown -R facecv:facecv /app

USER facecv

# Production command with workers
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "4"]