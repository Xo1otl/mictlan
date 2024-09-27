FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode

# cacheを活用するため、更新が少なく実行に時間がかかるものから順にRUNする
# nixはdirnevとflakeでフォルダ単位の環境スイッチ用
RUN curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install linux \
    --init none \
    --no-confirm && \
    sudo chown -R vscode:vscode /nix

RUN sudo apt update \
    && sudo apt install -y \
    direnv \
    fish \
    pipx \
    git-lfs \
    php \
    php-xdebug \
    php-mbstring \
    php-mysql \
    && pipx install poetry \
    && curl -fsSL https://bun.sh/install | bash \
    && curl -sS https://getcomposer.org/installer | sudo php -- --install-dir=/usr/local/bin --filename=composer \
    && curl -L https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz | tar xzf - -C ~/.local \
    && ln -s ~/.local/nvim-linux64/bin/nvim ~/.local/bin/nvim \
    && sudo curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && sudo unzip awscliv2.zip \
    && sudo ./aws/install \
    && sudo rm -rf aws/ awscliv2.zip

# 残りのパッケージはnixでインストールする、dotfilesに環境インストール用のflakeが含まれているので事前にコピーしておく
COPY --chown=vscode:vscode dotfiles/ /home/vscode
RUN /nix/var/nix/profiles/default/bin/nix profile install /home/vscode/.config/nix

# fishのパスに対応してないので手動でパスを通している
RUN curl -fsSL https://install.julialang.org | sh -s -- -y --add-to-path=no
