package digitguesser

import (
	"lib/pkg/ann"
	"math"

	"gonum.org/v1/gonum/mat"
)

type GonumContextData struct {
	Weights []*mat.Dense
	Biases  []*mat.VecDense
}

type GonumProcessor struct{}

func (g GonumProcessor) Activation(weightedSum ann.Number) ann.Number {
	// シグモイド関数
	return ann.Number(1 / (1 + math.Exp(-float64(weightedSum))))
}

func (g GonumProcessor) Derivative(weightedSum ann.Number) ann.Number {
	// シグモイド関数を微分したもの
	activation := g.Activation(weightedSum)
	return activation * (1 - activation)
}

func (g GonumProcessor) FeedForward(network *ann.Network, dataset ann.Dataset) []ann.Context[GonumContextData] {
	contexts := make([]ann.Context[GonumContextData], len(dataset))
	layers := network.Layers()
	connections := network.Connections()

	// ネットワークの重みと偏りを行列に変換
	weights := make([]*mat.Dense, len(layers)-1)
	biases := make([]*mat.VecDense, len(layers)-1)

	for i := 1; i < len(layers); i++ {
		layerConnections := connections[i]
		rows := len(layerConnections)
		cols := len(layerConnections[0])

		weightData := make([]float64, rows*cols)
		biasData := make([]float64, rows)

		for j, neuronConns := range layerConnections {
			for k, conn := range neuronConns {
				weightData[j*cols+k] = float64(conn.Weight())
			}
			biasData[j] = float64(layers[i].Neurons()[j].Bias())
		}

		weights[i-1] = mat.NewDense(rows, cols, weightData)
		biases[i-1] = mat.NewVecDense(rows, biasData)
	}

	// ContextDataの作成と格納
	contextData := GonumContextData{
		Weights: weights,
		Biases:  biases,
	}

	// データセットを行列に変換
	dataMatrix := mat.NewDense(len(dataset), len(dataset[0]), nil)
	for i, data := range dataset {
		for j, val := range data {
			dataMatrix.Set(i, j, float64(val))
		}
	}

	// フィードフォワード計算
	activations := make([]*mat.Dense, len(layers))
	activations[0] = dataMatrix

	for i := 1; i < len(layers); i++ {
		prevActivation := activations[i-1]
		weight := weights[i-1]
		bias := biases[i-1]

		weightedSum := mat.NewDense(prevActivation.RawMatrix().Rows, weight.RawMatrix().Rows, nil)
		weightedSum.Mul(prevActivation, weight.T())

		// バイアスの加算
		rows, cols := weightedSum.Dims()
		biasMatrix := mat.NewDense(rows, cols, nil)
		for r := 0; r < rows; r++ {
			biasMatrix.SetRow(r, bias.RawVector().Data)
		}
		weightedSum.Add(weightedSum, biasMatrix)

		// 活性化関数の適用
		activationFunc := func(_, _ int, v float64) float64 {
			return float64(g.Activation(ann.Number(v)))
		}
		weightedSum.Apply(activationFunc, weightedSum)

		activations[i] = weightedSum
	}

	// 結果をコンテキストに変換
	for i := range dataset {
		layerSizes := make([]int, len(layers))
		for i, layer := range layers {
			layerSizes[i] = len(layer.Neurons())
		}
		context := ann.NewContext[GonumContextData](layerSizes)
		context.Data = contextData
		for j, activation := range activations {
			layerActivation := make(ann.LayerActivations, activation.RawMatrix().Cols)
			for k := 0; k < activation.RawMatrix().Cols; k++ {
				layerActivation[k] = ann.Number(activation.At(i, k))
			}
			err := context.SetActivations(j, layerActivation)
			if err != nil {
				panic(err)
			}
		}
		contexts[i] = *context
	}

	return contexts
}

func (g GonumProcessor) BackPropagate(contexts []ann.Context[GonumContextData], labels []ann.Labels) ann.Adjustments {
	if len(contexts) == 0 || len(labels) != len(contexts) {
		panic("Invalid input: contexts and labels must have the same non-zero length")
	}

	numLayers := len(contexts[0].Activations())
	weights := contexts[0].Data.Weights
	batchSize := len(contexts)

	// Initialize weight and bias adjustments
	weightAdjustments := make([]ann.WeightAdjustmentsToLayer, numLayers-1)
	biasAdjustments := make([]ann.BiasAdjustmentsToLayer, numLayers-1)

	// Convert activations and labels to matrices
	activations := make([]*mat.Dense, numLayers)
	for l := 0; l < numLayers; l++ {
		layerSize := len(contexts[0].Activations()[l])
		activationData := make([]float64, batchSize*layerSize)
		for i, context := range contexts {
			for j, act := range context.Activations()[l] {
				activationData[i*layerSize+j] = float64(act)
			}
		}
		activations[l] = mat.NewDense(batchSize, layerSize, activationData)
	}

	labelsMatrix := mat.NewDense(batchSize, len(labels[0]), nil)
	for i, label := range labels {
		for j, val := range label {
			labelsMatrix.Set(i, j, float64(val))
		}
	}

	// Backpropagation
	delta := new(mat.Dense)
	delta.Sub(activations[numLayers-1], labelsMatrix)

	for l := numLayers - 2; l >= 0; l-- {
		prevLayerSize := activations[l].RawMatrix().Cols
		currentLayerSize := activations[l+1].RawMatrix().Cols

		// Calculate weight adjustments
		weightAdj := new(mat.Dense)
		weightAdj.Mul(activations[l].T(), delta)
		weightAdj.Scale(1/float64(batchSize), weightAdj)

		// FIXME: でかい行列用意してから、最後にまとめてやったほうが良い気がする
		weightAdjustments[l] = make(ann.WeightAdjustmentsToLayer, currentLayerSize)
		for i := 0; i < currentLayerSize; i++ {
			weightAdjustments[l][i] = make(ann.WeightAdjustmentsToNeuron, prevLayerSize)
			for j := 0; j < prevLayerSize; j++ {
				weightAdjustments[l][i][j] = ann.Number(weightAdj.At(j, i))
			}
		}

		// Calculate bias adjustments
		biasAdj := make(ann.BiasAdjustmentsToLayer, currentLayerSize)
		for i := 0; i < currentLayerSize; i++ {
			sum := 0.0
			for j := 0; j < batchSize; j++ {
				sum += delta.At(j, i)
			}
			biasAdj[i] = ann.BiasAdjustmentsToNeuron(sum / float64(batchSize))
		}
		biasAdjustments[l] = biasAdj

		if l > 0 {
			// Prepare delta for the next layer
			newDelta := new(mat.Dense)
			// FeedForwardの時にweightsが転地された状態になってる、statefulなオブジェクトのやり取りは注意が必要
			newDelta.Mul(delta, weights[l])

			// Element-wise multiplication with the derivative of the activation function
			derivativeActivation := new(mat.Dense)
			derivativeActivation.Apply(func(_, _ int, v float64) float64 {
				return float64(g.Derivative(ann.Number(v)))
			}, activations[l])

			delta = new(mat.Dense)
			delta.MulElem(newDelta, derivativeActivation)
		}
	}

	return ann.NewAdjustments(weightAdjustments, biasAdjustments)
}

func NewGonumProcessor() ann.Processor[GonumContextData] {
	return GonumProcessor{}
}
