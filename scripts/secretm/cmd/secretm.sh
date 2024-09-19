#!/bin/bash

script_dir=$(dirname $(readlink -f "$0"))
entrypoint="$script_dir/../internal/api.php"
php "$entrypoint" "$@"
