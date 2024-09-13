<?php
// 出力バッファリングを開始
ob_start();

// helloと出力
echo 'hello';

// 出力バッファをフラッシュしてクライアントに送信
ob_flush();
flush();

// 10秒間スリープ
sleep(10);

// その後の処理
echo '10 seconds have passed.';

// 出力バッファを終了
ob_end_flush();
