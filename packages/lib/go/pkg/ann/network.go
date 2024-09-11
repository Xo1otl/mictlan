package ann

import (
	"fmt"
	"math/rand/v2"
)

type Number float64

type Neuron struct {
	bias Number
}

func NewNeuron(bias Number) *Neuron {
	return &Neuron{bias}
}

func (n *Neuron) Bias() Number {
	return n.bias
}

func (n *Neuron) SetBias(bias Number) {
	n.bias = bias
}

func NewNeurons(count int, bias Number) []*Neuron {
	neurons := make([]*Neuron, count)
	for i := 0; i < count; i++ {
		neurons[i] = NewNeuron(bias)
	}
	return neurons
}

type Connection struct {
	from   *Neuron
	to     *Neuron
	weight Number
}

func NewConnection(from *Neuron, to *Neuron, weight Number) *Connection {
	return &Connection{from, to, weight}
}

func (c *Connection) Weight() Number {
	return c.weight
}

func (c *Connection) AdjustWeight(delta Number) {
	// FIXME: 重みの範囲を制限する必要があるかもしれない
	c.weight += delta
}

type Layer struct {
	neurons []*Neuron
}

func NewLayer(neurons []*Neuron) *Layer {
	return &Layer{neurons: neurons}
}

func (l *Layer) Neurons() []*Neuron {
	return l.neurons
}

// NeuronConnections はあるニューロンに対する前の層からの接続
type NeuronConnections []*Connection
type LayerConnections []NeuronConnections
type Connections []LayerConnections

type Network struct {
	layers []*Layer
	// FIXME: この形式だと代入がだるいので、connectionsの部分テンソル等を
	connections Connections
}

func NewNetwork(layers []*Layer, initialWeight ...Number) *Network {
	network := &Network{layers: layers}
	network.Connect(initialWeight...)
	return network
}

func (n *Network) Layers() []*Layer {
	return n.layers
}

func (n *Network) Connections() Connections {
	return n.connections
}

func (n *Network) Connect(initialWeight ...Number) {
	n.connections = make(Connections, len(n.layers))

	for toLayerIndex := 1; toLayerIndex < len(n.layers); toLayerIndex++ {
		fromLayer := n.layers[toLayerIndex-1]
		toLayer := n.layers[toLayerIndex]

		// LayerConnectionsの要素は、次の層のニューロン毎に1つ存在する接続の配列なので、次の層のニューロンの数準備する
		n.connections[toLayerIndex] = make(LayerConnections, len(toLayer.neurons))

		for toNeuronIndex, toNeuron := range toLayer.neurons {
			// neuronConnectionsは、次の層の各ニューロンが持つ接続なので、現在の層のすべてのニューロンの数準備する
			neuronConnections := make(NeuronConnections, len(fromLayer.neurons))

			for fromNeuronIndex, fromNeuron := range fromLayer.neurons {
				weight := Number(0)
				if initialWeight != nil {
					weight = initialWeight[0]
				} else {
					weight = Number(rand.Float64()*2 - 1)
				}
				connection := NewConnection(fromNeuron, toNeuron, weight)
				neuronConnections[fromNeuronIndex] = connection
			}

			n.connections[toLayerIndex][toNeuronIndex] = neuronConnections
		}
	}
}

// TODO: 引数の与え方これが最適化わからないけど何らかの方法で値をセットできるようにする

func (n *Network) AdjustNeuronConnections(layerIndex int, neuronIndex int, weightAdjustments []Number) error {
	if layerIndex < 0 || layerIndex >= len(n.connections) {
		return fmt.Errorf("invalid layer index: %d", layerIndex)
	}

	if neuronIndex < 0 || neuronIndex >= len(n.connections[layerIndex]) {
		return fmt.Errorf("invalid neuron index: %d for layer %d", neuronIndex, layerIndex)
	}

	conns := n.connections[layerIndex][neuronIndex]

	if len(weightAdjustments) != len(conns) {
		return fmt.Errorf("mismatch in number of adjustments (%d) and connections (%d)", len(weightAdjustments), len(conns))
	}

	for index, conn := range conns {
		conn.AdjustWeight(weightAdjustments[index])
	}

	return nil
}
