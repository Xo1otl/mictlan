# infra

infra

### 現在の仕組み

envは最初に読み込まれる

templateファイルはmoduleに属していない

composeは遅延ロードされる、AのcomposeでBのenvが必要、BのcomposeでAのenvが必要

composeは今のところ全部yamlのため、オブジェクト置いとけばそれをyamlにしている

templateはyamlだったりちがったりする、これは出力部分まで書くようにして、ただ実行しているだけである

### 構成要素

上をまとめると、現在のinfraの設定は二つに分類される

* 決まった変換がなされている設定
* 決まった変換がないから実行しているだけの設定

### 決まった変換がされている設定

大体はComposeだけどすべてそうとも限らないが、全部composeと考えても別によい気がする

### 決まった変換がされていない設定

nginxとかmysqlのinit scriptなどなど

### クラスにする案
```
from workspace import infra

infra.register
```

というようにしたいが、循環参照どうしよう

現在は遅延インポートする関数を集計している

registerは一か所で行いたい

現在遅延している部分としていない部分を、configとcomposeという風に二つのmoduleにする

つまりinfra一つ一つにつき二つずつmoduleが出現する形
