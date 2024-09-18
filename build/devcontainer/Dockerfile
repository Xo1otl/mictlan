FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode

RUN sudo apt update \
    && sudo apt install -y \
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

COPY --chown=vscode:vscode dotfiles/ /home/vscode

RUN echo "\nif [[ \$(ps -o comm= -p \$PPID) != \"fish\" ]]; then\n    exec fish\nfi" >> /home/vscode/.bashrc
