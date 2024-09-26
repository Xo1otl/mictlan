FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode

RUN curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install linux \
    --init none \
    --no-confirm && \
    sudo chown -R vscode:vscode /nix
