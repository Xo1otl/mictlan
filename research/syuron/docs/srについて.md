# 聞くことまとめ

機械学習がうまくいくかどうか関係なく、探索は必要

機械学習なくても探索自体はできて、リソース的に探索だけでも現実的な時間で結果を期待することはできる

探索アルゴリズムをメインに作って機械学習は同時並行でやっていこうかなと思う

手法自体、単体で論文になるようなテーマだし、実装や理論を考えてもだいぶ大変なタスク

今後の研究の方針でこれで良いのか質問したい

論文を片山先生にも確認してもらいたい、Adamしなければめっちゃショートカットできるから、その方向性が可能か確認したい

Adamする場合は時間かかりそう、 $\Lambda_0 = \Lambda / (1 + rz)$ ぐらいの式なら探索は容易だが、Adamで $r$ を特定とかしだすと時間がかかる。前の結果みて更新するので系列に実行する必要があるため並列化不可能

LLM-SRする前提で高速化すべき場所を洗い出したいし、相談もしたい

ルンゲクッタ法してる部分は、ドメイン内で $\kappa$ が一定なら解析的に解けるから不要になる
