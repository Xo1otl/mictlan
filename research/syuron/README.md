# syuron

シミュレーションデータを基にして設計空間を近似するのをサロゲートモデルというらしい
機械学習で逆設計
shgのシミュレーションはpythonで行う

# 参考文献
* https://chatgpt.com/share/67d2ab46-c6b0-800b-a3fb-1cf1c48c0878
* https://www.mdpi.com/2673-3269/5/1/9#:~:text=training%20an%20artificial%20neural%20network,in%20comparison%20with

# TODO
* 入出力ひっくり返してみる
* MNIST と同じ条件のデータ作ってみる
* https://web.wakayama-u.ac.jp/~ntakayuk/palmtree-j.htm (NN以外もいろいろあるっぽい)

# Modules

## mgoslt

## train
* gen_dataset
* train
    * model
    * hyper_params
* preprocess

# TODO
* デバイス構造を楽に生成するためのモジュール欲しい
* 出力の値を規格化するか整数にしてやらんと、今の教師データは1E-6とかのスケールだから学習しにくい
* 遺伝的アルゴリズムがいいっぽい
* GNN試したい
* 分布を学習するのではなく、変換効率の数字を学習する。この時、対応する波長も入力に含める
* 誤差関数やデータのpreprocessでフーリエ変換をしてみるのありかも

## Memo
* ドメイン数が100の時だけ、フーリエ級数展開が30個程度できれいにフィッティングできる

## その他・ウエムー
* グラフの正当性を検証する
* プロトン交換導波路を使うから作り方等考えとく
* デバイスの作り方の計画を立てる
* デバイス図のドメインの数を増やす
* 式の下に図が欲しい

## デバイスの設計

光源どうするか考える

4月に講習受ける

作製を同時並行で行う
