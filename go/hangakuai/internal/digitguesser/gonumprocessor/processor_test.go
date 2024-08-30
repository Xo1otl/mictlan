package gonum

import (
	"math/rand/v2"
	"pkg/ann"
	"testing"
)

func TestProcessor(t *testing.T) {
	inputSize := 784
	outputSize := 10

	createNetwork := func() *ann.Network {
		inputLayer := ann.NewLayer(ann.NewNeurons(inputSize, 0))
		hiddenLayer1 := ann.NewLayer(ann.NewNeurons(16, 0))
		hiddenLayer2 := ann.NewLayer(ann.NewNeurons(16, 0))
		outputLayer := ann.NewLayer(ann.NewNeurons(outputSize, 0))

		// weightをすべて0.1にしてネットワークを作成
		// TODO: weightの初期値を複数用意するか、他の方法で極小値に収束するのではなく最小値を探したい
		return ann.NewNetwork([]*ann.Layer{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer})
	}

	network := createNetwork()
	oldNetwork := createNetwork()

	// トレインデータを作成
	trainDataNum := 1000
	dataset := make(ann.Dataset, trainDataNum)
	allLabels := make([]ann.Labels, trainDataNum)
	for i := range dataset {
		dataset[i] = make(ann.Data, inputSize)
		for j := range dataset[i] {
			dataset[i][j] = ann.Number(rand.Float64()*2 - 1)
		}
		allLabels[i] = make(ann.Labels, outputSize)
		for j := range allLabels[i] {
			// 教師データが全部0.0 0.1 0.2 ...0.9
			allLabels[i][j] = ann.Number(float32(j) * 0.1)
		}
	}

	processor := NewProcessor()
	// TODO: 学習率を勾配の急さに併せて調節する
	learningRate := ann.Number(0.1)

	// 50回の学習ループ
	for epoch := 0; epoch < 1000; epoch++ {
		contexts := processor.FeedForward(network, dataset)
		adjustments := processor.BackPropagate(contexts, allLabels)

		// ネットワークの重みを調整
		for layerIndex, layerAdjustments := range adjustments.WeightAdjustments() {
			for neuronIndex, neuronAdjustments := range layerAdjustments {
				scaledAdjustments := make([]ann.Number, len(neuronAdjustments))
				for i, adj := range neuronAdjustments {
					scaledAdjustments[i] = -learningRate * adj
				}

				err := network.AdjustNeuronConnections(layerIndex+1, neuronIndex, scaledAdjustments)
				if err != nil {
					t.Errorf("Error adjusting neuron connections: %v", err)
				}
			}
		}

		// バイアスの調整
		for layerIndex, layerBiasAdjustments := range adjustments.BiasAdjustments() {
			for neuronIndex, biasAdjustment := range layerBiasAdjustments {
				neuron := network.Layers()[layerIndex+1].Neurons()[neuronIndex]
				scaledBiasAdjustment := -learningRate * ann.Number(biasAdjustment)
				newBias := neuron.Bias() + scaledBiasAdjustment
				neuron.SetBias(newBias)
			}
		}

		// 10エポックごとに進捗を表示
		if epoch%10 == 0 {
			t.Logf("Completed epoch %d", epoch)
		}
	}

	t.Log("Training completed - 50 epochs")

	testDataset := make(ann.Dataset, 1)
	for i := range testDataset {
		testDataset[i] = make(ann.Data, inputSize)
		for j := range testDataset[i] {
			testDataset[i][j] = ann.Number(rand.Float64()*2 - 1)
		}
	}

	// 何を入力しても教師データの0 0.1 0.2 ...0.9という形式に近い出力になるはず
	finalContexts := processor.FeedForward(network, testDataset)
	t.Logf("Final output after training: %+v", finalContexts[0].Activations()[len(finalContexts[0].Activations())-1])

	// トレーニングしていないモデルではでたらめな出力になる
	randomContexts := processor.FeedForward(oldNetwork, testDataset)
	t.Logf("Output of model before training: %+v", randomContexts[0].Activations()[len(randomContexts[0].Activations())-1])
}
