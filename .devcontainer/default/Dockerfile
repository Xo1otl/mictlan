FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode

# cacheを活用するため、更新が少なく実行に時間がかかるものから順にRUNする
# 言語関連はここでインストールしておく
RUN sudo apt update && \
    sudo apt install -y direnv fish git-lfs php php-xdebug php-mbstring php-mysql fonts-noto-cjk && \
    curl -fsSL https://bun.sh/install | bash && \
    mkdir -p /home/vscode/.local/bin && \
    curl -L https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz | tar xzf - -C ~/.local && \
    ln -s ~/.local/nvim-linux-x86_64/bin/nvim ~/.local/bin/nvim && \
    # phpだけホームディレクトリが絶対パスで書かないと解決できない
    curl -sS https://getcomposer.org/installer | php -- --install-dir=/home/vscode/.local/bin --filename=composer && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /home/vscode/.local/bin/uv python install 3.12

# 上でUSER指定してても、COPYはrootで行われるため明示的に権限を設定する
COPY --chown=vscode:vscode dotfiles/ /home/vscode
