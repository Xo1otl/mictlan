## actor

声優のための機能

**interfaceのみ**

## admin

管理者のための機能

**interfaceのみ**

## auth

認証情報をとってくるところ

**interfaceのみ**

## guest

未ログか、他人の情報を見ているときのactorのことをguestと定義するguestのための機能

**interfaceのみ**

## kernel

すべてのコードの統合を行う、autoloadやmiddleware設定、一部の依存性注入など

## mysqladapter

interfaceの実装のうちmysqlに対するものすべて一箇所にまとめる

コネクションとかそういうのが一元管理しやすいし、インフラの変更時も一元的にできるほうがいいので横割り

上のインターフェースの中でrepositoryが登場すればそれを実装

**実装のみ**

## middleware

middlewareとしての実装はここにまとめる

middleware自体が横断して使用されるものだし、役割分担の縦割りより横割りのほうが適切

**interfaceと実装**

## query

検索など、これも役割分担というよりいろんなデータをまとめていい感じにとってくる感じなので横割り

queryではフィールドの有無ぐらいはあってもいいけど、値の型チェックは不要
