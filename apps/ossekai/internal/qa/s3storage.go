package qa

import (
	"context"
	"fmt"
	"io"
	"lib/pkg/transaction"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/brianvoe/gofakeit/v7"
)

type S3Storage struct {
	client *s3.Client
	bucket string
}

func NewS3Storage() CommandStorage {
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatal(err)
	}
	// Amazon S3 クライアントを作成
	client := s3.NewFromConfig(cfg)

	return &S3Storage{
		client: client,
		bucket: "ossekaiqa",
	}
}

// Put implements CommandStorage.
func (s *S3Storage) Put(tx transaction.Transaction, object *Object) (*Attachment, error) {
	var objectKey string

	// ロールバック処理を定義
	transaction.WithRollback(tx, func() {
		if objectKey != "" {
			// S3からオブジェクトを削除
			_, err := s.client.DeleteObject(context.TODO(), &s3.DeleteObjectInput{
				Bucket: aws.String(s.bucket),
				Key:    aws.String(objectKey),
			})
			if err != nil {
				log.Printf("ロールバック中にオブジェクト %s の削除に失敗しました: %v", objectKey, err)
			} else {
				log.Printf("トランザクションがロールバックされました。オブジェクトを削除しました: %s", objectKey)
			}
		}
	})

	// オブジェクトキーを生成
	objectKey = gofakeit.UUID()

	// コンテンツのサイズを計測するためにCountingReaderを使用
	body := &CountingReadSeekerAt{Reader: object.Src}

	// PutObjectInputを準備
	input := &s3.PutObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(objectKey),
		Body:   body,
	}

	// オブジェクトをS3にアップロード
	_, err := s.client.PutObject(context.TODO(), input)
	if err != nil {
		return nil, fmt.Errorf("S3へのオブジェクトのアップロードに失敗しました: %w", err)
	}

	size := body.BytesRead
	mimeType := "application/octet-stream"

	attachment := &Attachment{
		Placeholder: object.Placeholder,
		Kind:        mimeType,
		Size:        size,
		ObjectKey:   ObjectKey(objectKey),
	}

	return attachment, nil
}

type CountingReadSeekerAt struct {
	Reader    io.ReadSeeker
	BytesRead int64
}

func (cr *CountingReadSeekerAt) Read(p []byte) (int, error) {
	n, err := cr.Reader.Read(p)
	cr.BytesRead += int64(n)
	return n, err
}

func (cr *CountingReadSeekerAt) Seek(offset int64, whence int) (int64, error) {
	return cr.Reader.Seek(offset, whence)
}

func (cr *CountingReadSeekerAt) ReadAt(p []byte, off int64) (n int, err error) {
	if readerAt, ok := cr.Reader.(io.ReaderAt); ok {
		return readerAt.ReadAt(p, off)
	}
	_, err = cr.Seek(off, io.SeekStart)
	if err != nil {
		return 0, err
	}
	return cr.Read(p)
}
