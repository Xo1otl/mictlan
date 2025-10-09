FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER root

# --- Developer Tools Installation ---
RUN apt-get update && apt-get install -y curl gpg && \
    # mise, gcloud, hashicorpのリポジトリ設定を追加
    curl -fsSL https://mise.jdx.dev/gpg-key.pub | gpg --dearmor -o /etc/apt/keyrings/mise-archive-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/mise-archive-keyring.gpg arch=amd64] https://mise.jdx.dev/deb stable main" > /etc/apt/sources.list.d/mise.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(grep -oP '(?<=UBUNTU_CODENAME=).*' /etc/os-release || lsb_release -cs) main" > /etc/apt/sources.list.d/hashicorp.list && \
    # パッケージのインストール
    apt-get update && \
    apt-get install -y \
    fish \
    git-lfs \
    fonts-noto-cjk \
    mise \
    google-cloud-cli \
    terraform && \
    # クリーンアップ
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=vscode:vscode dotfiles/ /home/vscode
