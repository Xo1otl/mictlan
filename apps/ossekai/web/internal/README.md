# internal

Coding Style 通りに infra を明確にして機能ごとに分けようかと思ったけど、どうやって機能で分ければいいかわからなかったため全部一緒にした

多分フロントエンド流の分け方があって、hook だの pages だの書く必要ありそう

どうせパッケージ名となるのは自身の所蔵するフォルダ名なので、どこに置かれてても機能とインフラを明らかにしてればいいから、react しか使わなさそうな場合だと module/react と毎回しなくても

reactmodule みたいにしとけば良さそう

例えば auth モジュールを考えてみても、SignIn 等のページを auth に置くか pages に置くかなどの迷いどころがあり、めんどいから全部同じとろこに入れている

# Memo

今のdebugではボタン押してからfetchだからいいけど、コンポーネントの表示のときにfetchする場合loaderを使ってwaterfallを防ぐ