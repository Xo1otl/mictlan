package mnist

import "testing"

import (
	"fmt"
)

func TestReadLabelFile(t *testing.T) {
	ll, err := ReadLabelFile("data/t10k-labels-idx1-ubyte.gz")
	if err != nil {
		t.Fatalf("read (%s)", err)
	}
	if len(ll) != 10000 {
		t.Errorf("unexpected count %d", len(ll))
	}
}

func TestReadImageFile(t *testing.T) {
	nrow, ncol, imgs, err := ReadImageFile("data/t10k-images-idx3-ubyte.gz")
	if err != nil {
		t.Fatalf("read (%s)", err)
	}
	if len(imgs) != 10000 {
		t.Errorf("unexpected count %d", len(imgs))
	}
	fmt.Printf("%d images, %dx%d format\n", len(imgs), nrow, ncol)
}

func TestLoad(t *testing.T) {
	train, test, err := Load("./data")
	if err != nil {
		t.Fatalf("load (%s)", err)
	}
	println(train.Count(), test.Count())
	println(train.Images[0])
}
