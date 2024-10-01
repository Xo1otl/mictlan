alias c="clear"
alias e="exit"
alias phs="python3 -m http.server"
alias phpd="php -dxdebug.mode=debug -dxdebug.start_with_request=yes -dxdebug.client_port=7003"

fish_vi_key_bindings

if status is-interactive
    # Commands to run in interactive sessions can go here
end

set -g fish_greeting

fish_add_path $HOME/.elan/bin
fish_add_path $HOME/.bun/bin
fish_add_path $HOME/.local/bin # user installed binaries
fish_add_path $HOME/.juliaup/bin # user installed binaries
fish_add_path /workspaces/mictlan/scripts # workspace scripts

direnv hook fish | source
direnv export fish | source

functions --copy cd standard_cd

function cd
  standard_cd $argv; and ls -a
end
