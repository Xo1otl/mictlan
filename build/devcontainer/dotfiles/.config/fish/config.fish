alias c="clear"
alias e="exit"
alias phs="python3 -m http.server"
alias phpd="php -dxdebug.mode=debug -dxdebug.start_with_request=yes -dxdebug.client_port=7003"

if status is-interactive
    # Commands to run in interactive sessions can go here
end

functions --copy cd standard_cd

function cd
  standard_cd $argv; and ls
end

set -g fish_greeting

fish_add_path $HOME/.elan/bin
fish_add_path $HOME/.bun/bin
fish_add_path $HOME/.local/bin # user installed binaries

fish_vi_key_bindings
