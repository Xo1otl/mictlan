FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

# Copy project files
COPY pyproject.toml.docker  pyproject.toml
COPY README.md ./
COPY src/ src/
COPY ui/ ui/

# Install dependencies
RUN uv sync

# Set environment variables
ENV PYTHONPATH=/app/src
ENV GRADIO_USER=qunasys

# Expose Gradio default port
EXPOSE 7860

# Run the Gradio application
CMD ["uv", "run", "python", "ui/main.py"]
