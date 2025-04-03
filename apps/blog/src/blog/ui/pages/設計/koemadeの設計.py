import streamlit as st

"""
# koemadeの設計を考える
声優管理サイト

* 要件がよくわかってないからまとめる
* 人に例えて会社を作るつもりでやってみる

## 列挙

メール送信による声優登録、音声投稿、認証、検索やクエリ、タグ登録、声優ランク制度、バン機能

管理者だけがsignupを行えるようにしたいという要望、普通のsignupのエンドポイントを作成して管理者だけがアクセスできるようにする

普通のsignupページではメアドとパスワードなど必要情報を入力して登録、メアドにメールが送信される

管理者に登録申請する部分と、実際の登録は全く別で、登録申請の時に入力された情報も、実際の登録時には不要

```sql
CREATE DATABASE IF NOT EXISTS koemade;

USE koemade;

CREATE TABLE IF NOT EXISTS signup_requests (
    id BINARY(16) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    furigana VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    tel VARCHAR(20) NOT NULL,
    bank_name VARCHAR(255) NOT NULL,
    branch_name VARCHAR(255) NOT NULL,
    account_number VARCHAR(20) NOT NULL,
    self_promotion TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    id BINARY(16) PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    status ENUM('active', 'banned', 'suspended') NOT NULL DEFAULT 'active',
    INDEX idx_username (username)
);

CREATE TABLE IF NOT EXISTS roles (
    id BINARY(16) PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS account_role (
    account_id BINARY(16) NOT NULL,
    role_id BINARY(16) NOT NULL,
    PRIMARY KEY (account_id, role_id),
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS actor_profiles (
    display_name VARCHAR(255) NOT NULL,
    actor_rank VARCHAR(255) NOT NULL,
    self_promotion TEXT,
    price INT NOT NULL,
    account_id BINARY(16) PRIMARY KEY NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS nsfw_options (
    allowed BOOLEAN NOT NULL,
    price INT NOT NULL,
    extreme_allowed BOOLEAN NOT NULL,
    extreme_surcharge INT NOT NULL,
    account_id BINARY(16) PRIMARY KEY NOT NULL,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS profile_images (
    account_id BINARY(16) PRIMARY KEY NOT NULL,
    mime_type VARCHAR(255) NOT NULL,
    size INTEGER NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS voices (
    id BINARY(16) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    account_id BINARY(16) NOT NULL,
    mime_type VARCHAR(255) NOT NULL,
    path TEXT NOT NULL,
    hash VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts (id) ON DELETE CASCADE,
    INDEX (account_id),
    UNIQUE (title, account_id)
);

CREATE TABLE IF NOT EXISTS tags (
    id BINARY(16) PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    category VARCHAR(255) NOT NULL,
    UNIQUE (name, category)
);

CREATE TABLE IF NOT EXISTS voice_tag (
    voice_id BINARY(16) NOT NULL,
    tag_id BINARY(16) NOT NULL,
    PRIMARY KEY (voice_id, tag_id),
    FOREIGN KEY (voice_id) REFERENCES voices (id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE
);
```

## ドメインツリー構築
* koemadeというサイトを作る、という最も抽象的で大きな目標を、再帰的に補題に分割する
* デタラメに分割するのではなく、責務の分離や目的意識を持って、センスのいい分割を行う
* ここも人に例えて組織図を作るようにするとやりやすい

### コマンドクエリ分離
* 考え方の指標となるため、分割を行う上で意識したい

#### コマンド
声優の登録やプロフィールの編集や音声のアップロードはコマンド

#### クエリ
検索や一覧表示はクエリ、声優のクエリ、音声のクエリ、声優と音声を一緒にクエリ、それぞれ検索

### イベント・ドリブン
* これも考え方の指標となる、組織の人々を冒険者、イベントストア及びブローカーを冒険者ギルドと捉えると考えやすい
* ギルド掲示板(ブローカー)は冒険者専用であることに注意(冒険者の作業の結果と、その続きが貼られるが、直接リクエストが来ることはない)

### ドメインツリー
* `auth`: 関係者の身分証確認を行う人(jwtのclaimsを検証・確認してドメインオブジェクトにする)
    * シンプルにミドルウェアとして400返したり変数用意するだけ、ほかからrequireして使う
    * こいつはステートレス
* `query`: コマンドと同じ名前空間にしないために分離
    * `guestservice`: 匿名・一般の利用者に対して情報提供する人
        * `voices`: 音声投稿を紹介する人
            * `search`: 一覧
            * 詳細
        * `profiles`: プロフィールを紹介する人
            * 一覧
            * 詳細
    * `actorservice`: 関係者に対して情報提供する人
        * `profile`: その人のプロフィールの詳細
        * `myvoices`: その人の音声の一覧
        * `voice`: その人の音声の詳細
* `actor`: 関係者の情報への処理
    * `profile`: その人のプロフィールを追加・編集
    * `voice`: その人の音声を追加・編集
* `iam`
    * アカウントのban
    * アカウント登録
    * アカウントの一覧
* `application`: 登録申請転送人
    * 登録申請があった時に運営に謎メール送りつけるだけのステートレスな謎サービス
    
### 主語+サービスという分離
akinatorの時みたいに、trainとかqaとか、そういう動詞を用いた機能的な分離が向いてないことに気づいた

それよりも、actorservice, guestservice, iam, applicationというように、主語+サービスという分離が向いている

actorserviceは再帰的に、profileservice, voiceserviceというように分離される

この〇〇サービスというのは、〇〇repositoryみたいな感じで、汎用性の高い表現として**標準化して良い**と思う

### CQRSの捉え方の変更
もともとドメイン設計でCQRSを意識し、完全に別なものとして考えようとしていた

よう考えたらakinatorでもqaがsend_answerなる関数を持ちコマンドもしていて、ドメインレベルでの分離できてなかった

しかし、CQRSはもっとシンプルに、追加と読み取りの間にviewという概念や変換を挟んで、データベース部分でわけるということだと思い始めた

つまりドメインはもっと自由にしてよく、書き込んだデータから形成可能な情報に関しては、理想的な形でデータがあることを前提としたさらなる自由な設計をすべきということだと思い始めた

つまりドメイン設計での指針というより、ドメインをより自由に考えるためのインフラの構成法という気がしてきた

CQRSとCRUDもハイブリッドで行くのがいいかもしれない

様々な分け方を知っておいて、その時々で最適な択を選択しよう！

CQRSは純粋に、読み取りたいデータについて、書き込んだデータの形式に引っ張られずに柔軟に考えて、もし必要ならばデータを変換して提供するという考え方だと思う

## フリーダム設計

いろいろ考えているうちに、repositoryという抽象化がだるい気がしてきた

ドメインレイヤは再帰的にインターフェースに分割され、それ以上分割しないインターフェースについて、適切な実装を行う

その中で、最終的なインターフェースの択としてrepositoryというものが存在する可能性があるだけであって、そんなに必要不可欠なものではない気がする

つまりProfileService！とかそういう名前で編集だの追加だのを行うインターフェース考えた時に、それが十分単純でわかりやすいならrepositoryいらん

強いて言うなら、queryの種類としてsearchとrepositoryという二種類に分けるのありな気がする

queryは、書き込みとごっちゃにならないように意識的に分けたほうがいい場合が多い

結論として、再帰的にインターフェースを用いた補題分割をしよう、それ以上分割できないインターフェースができたら、インフラ使った実装コードを別でかこう、それ以上の決まりはなく、すべて自由なのではないか

再帰的な補題への分割、末端のインターフェースをインフラを使用したadapterで実装、最後にDI、これが全てな気がする

### フォルダの切り方について

これも絶対たてわり絶対横割りとかなくて、ものによる説がでてきた

middlewareとどこでも必要なadapter等は横割りのほうがいい説がある

横割りと縦割りについてまとめよう
"""
