package main

import (
	"fmt"
	"hangauai/internal/digitguesser"
	gonum "hangauai/internal/digitguesser/gonumprocessor"
	"hangauai/internal/mnist"
	"log"
	"pkg/ann"
)

func main() {
	dataset, labels, err := mnist.LoadData("/workspaces/mictlan/go/hangakuai/internal/mnist/data/")
	if err != nil {
		log.Fatalf("Failed to load MNIST data: %v", err)
	}
	fmt.Printf("Dataset size: %d, Labels size: %d\n", len(dataset), len(labels))

	processor := gonum.NewProcessor()
	guesser := digitguesser.NewApp(processor, dataset[:60000], labels[:60000])

	// 学習の実行
	err = guesser.Train(2000)
	if err != nil {
		panic(err)
	}

	// テストデータでの予測と正答率の計算
	correct := 0
	total := 0
	for i := 60000; i < 70000; i++ {
		prediction := guesser.Predict(dataset[i])

		predictedIndex := maxIndex(prediction)
		actualIndex := maxIndex(labels[i])

		if predictedIndex == actualIndex {
			correct++
		}
		total++

		// 予測結果の表示（オプション）
		//fmt.Printf("Sample %d: Predicted %d, Actual %d, %v\n", i, predictedIndex, actualIndex, predictedIndex == actualIndex)
	}

	// 正答率の計算と表示
	accuracy := float64(correct) / float64(total) * 100
	fmt.Printf("\nAccuracy: %.2f%% (%d/%d)\n", accuracy, correct, total)
}

// maxIndex は与えられたスライスの中で最大値のインデックスを返す
func maxIndex(slice []ann.Number) int {
	maxVal := slice[0]
	maxIdx := 0
	for i, val := range slice {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}
