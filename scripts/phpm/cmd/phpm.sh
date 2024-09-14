#!/bin/bash

script_dir=$(dirname $(readlink -f "$0"))
entrypoint="$script_dir/../internal/phpm.php"
php "$entrypoint" "$@"
