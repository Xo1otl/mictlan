package ann

import "fmt"

type LayerActivations []Number

// Context はすべてのActivationを保持している
type Context[T any] struct {
	activations []LayerActivations
	Data        T
}

// NewContext creates a new Context with initialized activations
func NewContext[T any](layerSizes []int) *Context[T] {
	activations := make([]LayerActivations, len(layerSizes))
	for i, size := range layerSizes {
		activations[i] = make(LayerActivations, size)
	}
	return &Context[T]{activations: activations}
}

func (c *Context[T]) Activations() []LayerActivations {
	return c.activations
}

func (c *Context[T]) SetActivations(layerIndex int, activations LayerActivations) error {
	if layerIndex < 0 || layerIndex >= len(c.activations) {
		return fmt.Errorf("invalid layer index: %d", layerIndex)
	}
	if len(activations) != len(c.activations[layerIndex]) {
		return fmt.Errorf("mismatch in activations length: expected %d, got %d", len(c.activations[layerIndex]), len(activations))
	}
	c.activations[layerIndex] = activations
	return nil
}

// Data は入力層のニューロンの値の配列
type Data []Number
type Dataset []Data

type WeightAdjustmentsToNeuron []Number
type WeightAdjustmentsToLayer []WeightAdjustmentsToNeuron
type WeightAdjustments []WeightAdjustmentsToLayer

type BiasAdjustmentsToNeuron Number
type BiasAdjustmentsToLayer []BiasAdjustmentsToNeuron
type BiasAdjustments []BiasAdjustmentsToLayer

type Adjustments struct {
	weightAdjustments WeightAdjustments
	biasAdjustments   BiasAdjustments
}

func NewAdjustments(adjustments WeightAdjustments, biasAdjustments BiasAdjustments) Adjustments {
	return Adjustments{adjustments, biasAdjustments}
}

func (a *Adjustments) WeightAdjustments() WeightAdjustments {
	return a.weightAdjustments
}

func (a *Adjustments) BiasAdjustments() BiasAdjustments {
	return a.biasAdjustments
}

// ActivationCalculator は活性化関数と活性化関数の導関数の値を計算するための構造体のインターフェース
type ActivationCalculator interface {
	// Activation は活性化関数の値
	Activation(weightedSum Number) Number
	// Derivative は導関数の値
	Derivative(weightedSum Number) Number
}

// Labels は教師データ
type Labels []Number

// Processor では行列計算を行う必要があるためニューロン毎に違うActivationを得ることは不可能
type Processor[T any] interface {
	ActivationCalculator
	FeedForward(network *Network, dataset Dataset) []Context[T]
	BackPropagate(contexts []Context[T], labels []Labels) Adjustments
}
