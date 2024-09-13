<?php

class User
{
    public int $id;
    public string $name;
    public string $email;

    public function __construct($id, $name, $email)
    {
        $this->id = $id;
        $this->name = $name;
        $this->email = $email;
    }
}

// オブジェクトの作成
$user = new User(1, 'John Doe', 'john.doe@example.com');

// オブジェクトをシリアライズ
$serializedUser = serialize($user);

// シリアライズされたオブジェクトをファイルに書き出し
file_put_contents('user_data.txt', $serializedUser);

// ファイルからシリアライズされたデータを読み込み
$serializedUserFromFile = file_get_contents('user_data.txt');

// データをアンシリアライズしてオブジェクトに再格納
$unserializedUser = unserialize($serializedUserFromFile);

// 結果の確認
echo "ID: " . $unserializedUser->id . "\n";
echo "Name: " . $unserializedUser->name . "\n";
echo "Email: " . $unserializedUser->email . "\n";
