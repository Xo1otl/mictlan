# php

phpのワークスペース

# Note

jupyter起動してる時php実行したらxdebugがjupyterに接続しようとして固まる
`php -n -d zend_extension=/usr/local/php/8.3.11/extensions/xdebug.so -d xdebug.mode=off <filename>`
でいける