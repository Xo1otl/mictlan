import streamlit as st

"""
# 設計について

設計といえばClean ArchitectureとDDD、あとCQRSとEvent Sourcingをかじりました

**ここに書いているのはめちゃくちゃ持論です**

## Clean Architecture

### Layerは3つが必要十分、ドメインとインフラとアダプタ

増やすと逆効果で減らすと成立しません

#### ドメイン
ドメイン書く時はインフラ用語使用禁止です。よく学校とかで、自分の言葉で説明しろ！と言われるあれと同じぐらいインフラ用語をつかうのはNGです

最悪自分で造語を作って、インフラ用語を使うのを避けましょう、紛らわしいですが、RFCの用語はドメインで使ってもいいです

Clean Architectureにおけるドメインを直訳すると、**目的**になると思っています

#### インフラ
フレームワークもここに含まれます、ここは自分では書かない場合が多いです

自分で書くのはUIぐらい

#### アダプタ
インフラを使って、ドメインで書いたインターフェースを実装する部分です

ここを書く時は、ドメインが完成してインフラの選定も終わってる時なので、関数の入出力の型と使用するライブラリまでわかっている状態なので、アダプタの作成は結構作業です、ほぼGPTがやってます

インフラが無い時はMockを書いてアダプタ内で完結させます

実際書く順番的には ドメイン -> アダプタ(Mock) -> インフラ準備 -> アダプタ(実装) みたいな感じかもしれん

### 自分が王様だと思って設計をするとやりやすい

ドメインが仕様書、インフラが民衆、アダプタが中間管理職です

まずわがままを言います。ここでは誰がなにをするのかまでは考えず、とりあえずあれがしたいこれがしたいという感じで決めます

次にインフラを用意します、王様のわがままを実現できそうな能力を持った人を沢山集めてきます

次が中間管理職の出番です、誰をどう組み合わせれば王様のわがままが実現できるか考えます

王様は民衆がどうやって自分のわがままを実現するかは一切知りません、下々のことなど考えるだけ時間の無駄である (RFCは上級国民です)

## DDD

自分はドメインの設計にだけ焦点をあててDDDを考えています

王様は国を良くするためにセンスのいいわがままを言う必要があります

自分はこれを補題への再帰的な分割と捉えています

他にも入出力のデータの形から考えていくという手法が一般的で、IDを持つデータをEntity,IDを持たないものをValueObjectと呼びます

ロジックは単純だけど考えないといけないことが多い時はDomainやEntityを考えて、作り方考えるのが普通に難しい場合は補題分割を頑張ったらいいと思います

## Event Driven Architecture

この**Event**というのは処理の結果を表す事後イベントです

EDAはギルドを考えるとわかりやすいと思っています

brokerがギルド掲示板で、consumerが冒険者で、掲示板に貼ってあるクエストを見て自分にできるクエストを選んでやる感じです

また、冒険者はできるところだけやってギルドに残りの作業をproduce(丸投げ)することもあります

あと、brokerでhttpリクエストをちょくで受け取ることはしません、つまり依頼は冒険者がまず受け取ってやることやった後、続きの作業をギルドに投げるという感じです

つまりこのギルド掲示板は一般人が勝手に依頼できるものではなく、冒険者同士でやり取りするためのものです

当然それぞれの冒険者はClean Architectureで設計するので、それぞれの冒険者が自分の国を持ってる王様という感じです

## CQRS

CQRSは、書き込みデータと成果物を別々のデータとして扱うことを表しますが、これは当たり前のことだと思っています

多分この文章読んでたら書き込むときのデータと成果物のデータが同じわけないだろと思うかもしれないですが、チュートリアルでありがちなTODOアプリとかで、
作成時の入力と実際の画面に表示されるデータが一緒すぎるせいで、作ったデータがそのまま読めるみたいな特殊な事例を普通だと思いがちなのかもしれないなと思っています

実際は使い勝手とか検索機能とかAnalyticsとか考えると、書き込んだデータそのままで十分なはずがなく、検索エンジンとかベクトルデータベースとかviewに形を変えて保存します

## Event Sourcing

イベントソーシングは、過去から現在までに至るまでのすべてのイベントを記録し、それらの影響をすべて集計して我々に見えるデータを形作るという設計です

プログラムでこれを実装するのはかなり大変なんですが、人間はイベントソーシングを意識すべきだと思っていて、起こったことを日記やブログや写真ですべて記録して置くことで、projectionの幅が広がると思っています

物理的にイベントソーシングしたら家がゴミ屋敷になりますが、特にテキストデータなどの情報というのはとてもデータが小さいため、一生のうちに考えたすべての内容をイベントとして記録しても、普通にprojectionできると思っています

なので自分のmonorepoでは、書いたものは消さずにすべて残すということを心がけて、その集大成として新しいものを作って行こうという感じです
"""
