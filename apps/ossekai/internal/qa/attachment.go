package qa

type Attachment struct {
	Placeholder string
	Kind        string
	Size        int64
	ObjectKey   ObjectKey
}

func NewAttachment(placeholder string, kind string, size int64, objectKey ObjectKey) *Attachment {
	return &Attachment{
		Placeholder: placeholder,
		Kind:        kind,
		Size:        size,
		ObjectKey:   objectKey,
	}
}

type Attachments []*Attachment
