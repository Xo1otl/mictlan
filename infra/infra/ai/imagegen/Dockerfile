FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/ComfyUI && \
    git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI && \
    python -m venv --system-site-packages /app/ComfyUI/venv && \
    . /app/ComfyUI/venv/bin/activate && \
    pip install --no-cache-dir -r /app/ComfyUI/requirements.txt && \
    cd /app/ComfyUI/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    deactivate

RUN mkdir -p /app/fluxgym && \
    git clone https://github.com/cocktailpeanut/fluxgym.git /app/fluxgym && \
    cd /app/fluxgym && \
    git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git && \
    python -m venv --system-site-packages /app/fluxgym/venv && \
    . /app/fluxgym/venv/bin/activate && \
    cd sd-scripts && \
    pip install --no-cache-dir -r requirements.txt && \
    cd .. && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir voluptuous && \
    deactivate

ENV GRADIO_SERVER_NAME="0.0.0.0"
