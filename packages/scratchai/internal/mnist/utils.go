package mnist

import (
	"lib/pkg/ann"
	"path"
)

// Set represents a data set of image-label pairs held in memory
type Set struct {
	NRow   int
	NCol   int
	Images []RawImage
	Labels []Label
}

// ReadSet reads a set from the images file iname and the corresponding labels file lname
func ReadSet(iname, lname string) (set *Set, err error) {
	set = &Set{}
	if set.NRow, set.NCol, set.Images, err = ReadImageFile(iname); err != nil {
		return nil, err
	}
	if set.Labels, err = ReadLabelFile(lname); err != nil {
		return nil, err
	}
	return
}

// Count returns the number of points available in the data set
func (s *Set) Count() int {
	return len(s.Images)
}

// Get returns the i-th image and its corresponding label
func (s *Set) Get(i int) (RawImage, Label) {
	return s.Images[i], s.Labels[i]
}

// Sweeper is an iterator over the points in a data set
type Sweeper struct {
	set *Set
	i   int
}

// Next returns the next image and its label in the data set.
// If the end is reached, present is set to false.
func (sw *Sweeper) Next() (image RawImage, label Label, present bool) {
	if sw.i >= len(sw.set.Images) {
		return nil, 0, false
	}
	return sw.set.Images[sw.i], sw.set.Labels[sw.i], true
}

// Sweep creates a new sweep iterator over the data set
func (s *Set) Sweep() *Sweeper {
	return &Sweeper{set: s}
}

// Load reads both the training and the testing MNIST data sets, given
// a local directory dir, containing the MNIST distribution files.
func Load(dir string) (train, test *Set, err error) {
	if train, err = ReadSet(path.Join(dir, "train-images-idx3-ubyte.gz"), path.Join(dir, "train-labels-idx1-ubyte.gz")); err != nil {
		return nil, nil, err
	}
	if test, err = ReadSet(path.Join(dir, "t10k-images-idx3-ubyte.gz"), path.Join(dir, "t10k-labels-idx1-ubyte.gz")); err != nil {
		return nil, nil, err
	}
	return
}

func LoadData(dir string) (dataset ann.Dataset, labels []ann.Labels, err error) {
	train, test, err := Load(dir)
	if err != nil {
		return nil, nil, err
	}

	// Combine train and test sets
	totalSize := train.Count() + test.Count()
	dataset = make(ann.Dataset, totalSize)
	labels = make([]ann.Labels, totalSize)

	index := 0

	// Process training data
	for i := 0; i < train.Count(); i++ {
		image, label := train.Get(i)
		dataset[index] = imageToData(image)
		labels[index] = labelToLabels(label)
		index++
	}

	// Process test data
	for i := 0; i < test.Count(); i++ {
		image, label := test.Get(i)
		dataset[index] = imageToData(image)
		labels[index] = labelToLabels(label)
		index++
	}

	return dataset, labels, nil
}

// Helper function to convert RawImage to ann.Data
func imageToData(image RawImage) ann.Data {
	data := make(ann.Data, len(image))
	for i, pixel := range image {
		data[i] = ann.Number(pixel) / 255.0 // Normalize pixel values to [0, 1]
	}
	return data
}

// Helper function to convert Label to ann.Labels
func labelToLabels(label Label) ann.Labels {
	labels := make(ann.Labels, 10) // 10 possible digits (0-9)
	labels[label] = 1.0            // Set the correct label to 1.0, others remain 0.0
	return labels
}
