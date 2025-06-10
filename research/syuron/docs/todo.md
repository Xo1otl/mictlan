# TODO

* 積分の$\Phi$の整合部分以外並列計算できそうなので修正する
    * phase_mismatch関数がzをfloatで取っている部分を修正する必要がある (このままだと並列化ができない)
    * 並列化手順
        1. 開始位置をcumsumなどで計算
        2. 開始位置と幅情報があれば各層のSH発生量を位相のずれの補正こみですべて並列計算できる
        3. この並列計算のためには、phase_mismatch関数がzをjnp.ndarrayでとれる必要性がある
        4. 現在、数値積分を前提にしてzがfloatで渡される仕様になっている
        5. さらに、温度と波長の条件を振れるように計算結果がテンソルとなっている
        6. meshgridで作ったテンソルを使う仕様になっているがzがfloatであるようなphase_mismatch関数を、zに対しても並列化できる必要がある
        7. そもそも幅の数って固定じゃないからどうしよう。vmap使うか迷う
* 現在の条件のままスペクトル分布をフラットにできそうな条件をさがす

# 参考文献
* [3セグの論文](https://opg.optica.org/ol/fulltext.cfm?uri=ol-23-24-1880&id=37050)
* [2セグはこれが最適](https://www.sciencedirect.com/science/article/pii/S0030402620313024)
* [教授が言ってたチャープとduty比の組み合わせ](https://www.ursi.org/proceedings/procGA08/papers/D02ap5.pdf)
