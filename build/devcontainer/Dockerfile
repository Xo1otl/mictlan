FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode

# cacheを活用するため、更新が少なく実行に時間がかかるものから順にRUNする
# 言語関連はここでインストールしておく
RUN sudo apt update && \
    sudo apt install -y direnv fish pipx git-lfs php php-xdebug php-mbstring php-mysql && \
    pipx install poetry && \
    curl -fsSL https://bun.sh/install | bash && \
    curl -L https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz | tar xzf - -C ~/.local && \
    ln -s ~/.local/nvim-linux64/bin/nvim ~/.local/bin/nvim && \
    # phpだけホームディレクトリが絶対パスで書かないと解決できない
    curl -sS https://getcomposer.org/installer | php -- --install-dir=/home/vscode/.local/bin --filename=composer && \
    # fishのパスに対応してないので手動でパスを通している
    curl -fsSL https://install.julialang.org | sh -s -- -y --add-to-path=no

COPY --chown=vscode:vscode dotfiles/ /home/vscode
