# auth

callback を受け取って subscribe できるように作るともっと良いかもしれない

けど純粋な state machine にそんなものいらない説もある

clean architecture 的に subscribe の実装は、微妙に外部依存感もあって迷う

StateMachine は、状態と作用をとって新しい状態を返すオブジェクト

内部に state machine を持っている
