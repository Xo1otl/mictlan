import streamlit as st

"""
# 理論を説明する方法

物事をわかりやすく説明する一般的な方法

AならばBであるという証明は、Aという型を受け取った時にBという型を返す関数と同形である(カリー=ハワード対応)

以下命題とその証明のセットのことを関数と呼ぶ

相手に説明するとは、自身の持つ関数を相手に送信することである

## アルゴリズム

十分な粒度で関数を再帰的に補題(blueprint)に分轄し、rootから順に補題に対して以下のステップを行う

一つ下のレベルの補題がすべて示されている仮定のもと枝を実装するステップが相手にスムーズに伝わる粒度を、十分な粒度とする

1. 命題の入力、出力、境界条件を明示する (境界条件も厳密には入力の一部だが人間は分けて考えた方がわかりやすい)
2. 命題の具体例、つまり入力と出力として実際の値を与えたものを示す (命題の中身には触れず、このような入力に対してこのような出力が示されますといった例を取る)
3. 言及されている補題をrootとする補題ツリーにおける、一つ下のレベルのすべての補題について以下を行う
    1. 補題の入力、出力、境界条件を明示する
    2. 命題の時と同様に補題の具体例を述べる
    3. 命題とすべての補題を踏まえた現在の位置を再確認する
4. それらの補題が示されている仮定のもと、現在の補題を実装する
    1. 論理や仮説演繹法を用いた純粋な証明か、例えによる納得でよい、こちらの関数を相手に送信できればよく、厳密さは問わない
5. 関数の一致度の確認を行う、これは入出力、境界条件、処理の3つの観点から、仮説演繹法によりおこなう
    1. 正しい入力例、出力例を学習者が推論し、実際にあっていれば入出力の型の送信結果を信頼する
    2. 境界条件についても同様に確認
    3. 命題を用いて計算される入出力ペアの例のうち、入力だけを学習者に与えて学習者に出力を計算してもらい、十分一致すれば学習者の実施した処理を信頼する

1~5をすべてのノードに対して行う

# 例
[この方法でテンソルの実装方法説明してみた](テンソルの実装)
"""
