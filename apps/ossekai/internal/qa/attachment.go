package qa

type Attachment struct {
	Placeholder string
	Type        string
	Size        int64
	ObjectKey   ObjectKey
}
type Attachments []Attachment
