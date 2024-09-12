package qa

import "context"

type Repo interface {
	AddQuestion(q Question)
	Answers(q Question) []Answer
}

type Storage interface {
	// ObjectKeyの生成はだいたいUUID等が必要だけどこれをapplication layerで行うにはgoogle/uuidなどの抽象化が必要
	// これはめんどくさすぎるけど、ストレージの実装はinfraを使用できるレイヤのためそこで行えば抽象化は不用
	// そのため、Storageはuuidを引数で受け取らない。これはawsのs3のPutObjectとは仕様が異なる
	PutObjects(ctx context.Context, objects Objects) (ObjectKey, error)
}
