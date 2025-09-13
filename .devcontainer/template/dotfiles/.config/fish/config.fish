alias c="clear"
alias e="exit"
alias phs="python3 -m http.server"

fish_vi_key_bindings

set -g fish_greeting

fish_add_path $HOME/.elan/bin
fish_add_path /workspaces/mictlan/scripts

function cd
  builtin cd $argv; and ls -a
end

mise activate fish | source
