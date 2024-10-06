FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode

# cacheを活用するため、更新が少なく実行に時間がかかるものから順にRUNする
RUN sudo apt update && \
    sudo apt install -y direnv fish pipx git-lfs php php-xdebug php-mbstring php-mysql && \
    pipx install poetry && \
    curl -fsSL https://bun.sh/install | bash && \
    curl -sS https://getcomposer.org/installer | sudo php -- --install-dir=/usr/local/bin --filename=composer && \
    curl -L https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz | tar xzf - -C ~/.local && \
    ln -s ~/.local/nvim-linux64/bin/nvim ~/.local/bin/nvim && \
    sudo curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    sudo unzip awscliv2.zip && \
    sudo ./aws/install && \
    sudo rm -rf aws/ awscliv2.zip && \
    curl -LO https://github.com/redpanda-data/redpanda/releases/latest/download/rpk-linux-amd64.zip && \
    mkdir -p ~/.local/bin && \
    unzip rpk-linux-amd64.zip -d ~/.local/bin/ && \
    rm rpk-linux-amd64.zip

COPY --chown=vscode:vscode dotfiles/ /home/vscode

# fishのパスに対応してないので手動でパスを通している
RUN curl -fsSL https://install.julialang.org | sh -s -- -y --add-to-path=no
