FROM mictlan-devcontainer:latest

USER root

# --- CUDA Toolkit Installation ---
# TODO: 環境に合わせた自動化を検討する
RUN apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit && \
    # クリーンアップ
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
