package digitguesser

import (
	"math/rand"
	"pkg/ann"
)

const (
	InputSize           = 28 * 28
	HiddenLayer1Size    = 16
	HiddenLayer2Size    = 16
	OutputLayerSize     = 10
	BatchSize           = 1000
	InitialLearningRate = ann.Number(5)
	MinLearningRate     = 0.02
	LearningRateDecay   = 0.99
)

type App[T any] struct {
	network   *ann.Network
	processor ann.Processor[T]
	dataset   ann.Dataset
	labels    []ann.Labels
}

func NewApp[T any](processor ann.Processor[T], dataset ann.Dataset, labels []ann.Labels) *App[T] {
	// レイヤーを作成
	inputLayer := ann.NewLayer(ann.NewNeurons(InputSize, 0.1))
	hiddenLayer1 := ann.NewLayer(ann.NewNeurons(HiddenLayer1Size, 0.1))
	//hiddenLayer2 := ann.NewLayer(ann.NewNeurons(HiddenLayer2Size, 0.1))
	outputLayer := ann.NewLayer(ann.NewNeurons(OutputLayerSize, 0.1))

	// ネットワークを作成
	network := ann.NewNetwork([]*ann.Layer{inputLayer, hiddenLayer1, outputLayer})

	return &App[T]{
		network:   network,
		processor: processor,
		dataset:   dataset,
		labels:    labels,
	}
}

func (a *App[T]) Train(epochs int) error {
	learningRate := InitialLearningRate

	for epoch := 0; epoch < epochs; epoch++ {
		// ミニバッチの作成
		batchData, batchLabels := a.createMiniBatch()

		// フィードフォワードとバックプロパゲーション
		contexts := a.processor.FeedForward(a.network, batchData)
		adjustments := a.processor.BackPropagate(contexts, batchLabels)

		// ネットワークの更新
		err := a.updateNetwork(adjustments, learningRate)
		if err != nil {
			return err
		}

		// 学習率の更新
		learningRate = a.updateLearningRate(learningRate, epoch)
	}

	return nil
}

func (a *App[T]) createMiniBatch() (ann.Dataset, []ann.Labels) {
	indices := rand.Perm(len(a.dataset))[:BatchSize]
	batchData := make(ann.Dataset, BatchSize)
	batchLabels := make([]ann.Labels, BatchSize)

	for i, idx := range indices {
		batchData[i] = a.dataset[idx]
		batchLabels[i] = a.labels[idx]
	}

	return batchData, batchLabels
}

func (a *App[T]) updateNetwork(adjustments ann.Adjustments, learningRate ann.Number) error {
	// 重みの更新
	for layerIndex, layerAdjustments := range adjustments.WeightAdjustments() {
		for neuronIndex, neuronAdjustments := range layerAdjustments {
			scaledAdjustments := make([]ann.Number, len(neuronAdjustments))
			for i, adj := range neuronAdjustments {
				scaledAdjustments[i] = -learningRate * adj
			}
			err := a.network.AdjustNeuronConnections(layerIndex+1, neuronIndex, scaledAdjustments)
			if err != nil {
				return err
			}
		}
	}

	// バイアスの更新
	for layerIndex, layerBiasAdjustments := range adjustments.BiasAdjustments() {
		for neuronIndex, biasAdjustment := range layerBiasAdjustments {
			neuron := a.network.Layers()[layerIndex+1].Neurons()[neuronIndex]
			scaledBiasAdjustment := -learningRate * ann.Number(biasAdjustment)
			newBias := neuron.Bias() + scaledBiasAdjustment
			neuron.SetBias(newBias)
		}
	}

	return nil
}

func (a *App[T]) updateLearningRate(currentRate ann.Number, epoch int) ann.Number {
	newRate := currentRate * LearningRateDecay
	if newRate < MinLearningRate {
		return MinLearningRate
	}
	return newRate
}

func (a *App[T]) Predict(input ann.Data) ann.LayerActivations {
	dataset := make(ann.Dataset, 1)
	dataset[0] = input
	context := a.processor.FeedForward(a.network, dataset)[0]
	return context.Activations()[len(context.Activations())-1]
}
