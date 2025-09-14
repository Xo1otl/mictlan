alias c="clear"
alias e="exit"
alias phs="python3 -m http.server"

fish_vi_key_bindings

set -g fish_greeting

fish_add_path $HOME/.elan/bin # leanは拡張機能の指示でインストールされる
fish_add_path /usr/local/cuda-13.0/bin # cuda-toolkitはここにインストールされる

function cd
  builtin cd $argv; and ls -a
end

mise activate fish | source
