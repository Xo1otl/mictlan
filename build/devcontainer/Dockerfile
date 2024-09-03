FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode

RUN sudo apt update \
    && sudo apt install -y \
    fish \
    vim \
    pipx \
    git-lfs \
    php \
    php-xdebug \
    php-mysql \
    && pipx install poetry \
    && curl -fsSL https://bun.sh/install | bash \
    && curl -sS https://getcomposer.org/installer | sudo php -- --install-dir=/usr/local/bin --filename=composer

COPY --chown=vscode:vscode dotfiles/ /home/vscode

RUN echo "\nif [[ \$(ps -o comm= -p \$PPID) != \"fish\" ]]; then\n    exec fish\nfi" >> /home/vscode/.bashrc
