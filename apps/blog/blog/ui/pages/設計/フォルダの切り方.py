import streamlit as st

"""
# フォルダの切り方

ドメインを考えて、目的別に切り分ける

基本的にverticalでmodularに切り分ける
"""

"""
---
## adapterの切り方
adapterの目的を考えた時に、例えばrepositoryをmysqlで実装したい場合、mysqlというフォルダに書くか、それぞれのmoduleでmysql_repositoryというのを書くかで迷う
例えばActorというモジュールがある時、mysql/ActorRepositoryの場合とactor/MysqlRepsitoryの2つの案がある
## mysql/ActorRepository
### Pros
* mysqlで分けた場合、connectionの共有がしやすい
* 後から見返した時、実装がまとまった場所にあると割とわかりやすい
* インターフェースを交換する場合、adapterをまとめて変更する必要がある場合が多く、moduleごとに別れていると探しにくい
### Cons
* mysqlの場合は考えやすいが、複数のインフラが融合した実装や一部でしか行わない実装などはmoduleごとに分けた方がわかりやすい
## actor/MysqlRepository
* 書きにくかった
## 結論
mysqlの場合はmysql/ActorRepositoryのように分ける、一部でしか使用されなかったり、複数のインフラに依存していて分けにくいものはmoduleごとに書く

---

## moduleのタイプ
* 目的別のmodule: 普通に機能ごとに別れたモジュール(多用されないインフラのadapterも含む)
    * 例: train, qa等
* 一部のadapter: (データベースなど特定のインフラを多用する場合、adapterを一つのモジュールにまとめるほうがいい場合がある)
    * 例: mysql/ActorRepository, mysql/TagRepository等
* entrypoint: 機能をまとめたり、middlewareを登録する場所
    * apiserver等
* middleware: entrypointで登録されるようなもの、auth等
    * 例: auth, layout等
    
middlewareと使用箇所の多いadapterは横割り、それ以外は縦割りがいいかなと思う
"""
