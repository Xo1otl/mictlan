package qa

type PlaceHolder string

// TODO: 検証が必要ならガッツリvalueObjectにするかも、でもそこまでしているサービスもあんま見かけないから不要かも
type Object struct {
	Name string
	Data []byte
}

type Objects map[PlaceHolder]Object

type ObjectKey string

// StorageKeyがファイル追加後に生成されるためバックエンドロジックでプレースホルダーと実際のファイルとの紐付けを行う必要がある
type Attachment struct {
	Name      string
	Type      string
	Size      int64
	ObjectKey ObjectKey
}
type Attachments map[PlaceHolder]Attachment
