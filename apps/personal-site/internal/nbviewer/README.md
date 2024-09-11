# notebook

notebook は引数とったりしない系のファイルを表示するモジュール

引数とったりしない点でテンプレートエンジンとは異なる

typst と jupyter に対応する予定だけど実装は presenter 層で行う

今ノートの種類とか指定してないのでライブラリでノートの判別できず presenter で種類に応じた処理もかけない状態なので Application 層を改良してから実装を合わせる

# TODO

presenter が render するのに時間がかかる場合がある

render 結果を返すだけの現在の仕様を考え直す必要がある
