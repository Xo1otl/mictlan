FROM mictlan-devcontainer:latest

USER vscode

# 機械学習関連
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit \
    && curl -fsSL https://ollama.com/install.sh | sh
