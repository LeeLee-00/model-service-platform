FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
# Install Python 3.12 and pip
RUN apt-get update && apt-get install -y software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python and pip alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt --ignore-installed

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

# Create models directory
RUN mkdir /models && \
    chown -R appuser:appuser /models

USER appuser

FROM base AS dev

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS prod

COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]