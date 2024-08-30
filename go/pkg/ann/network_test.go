package ann

import (
	"testing"
)

func TestNetwork(t *testing.T) {
	// レイヤーを作成
	inputLayer := NewLayer(NewNeurons(28*28, 0))
	hiddenLayer1 := NewLayer(NewNeurons(16, 0.1))
	hiddenLayer2 := NewLayer(NewNeurons(16, 0.1))
	outputLayer := NewLayer(NewNeurons(10, 0.3))

	// ネットワークを作成
	network := NewNetwork([]*Layer{inputLayer, hiddenLayer1, hiddenLayer2, outputLayer})

	// テスト: レイヤー数が正しいか
	if len(network.Layers()) != 4 {
		t.Errorf("Expected 3 layers, got %d", len(network.Layers()))
	}

	// テスト: ニューロン数が各レイヤーで正しいか
	if len(network.Layers()[0].neurons) != 784 {
		t.Errorf("Expected 784 neurons in input layer, got %d", len(network.Layers()[0].neurons))
	}
	if len(network.Layers()[1].neurons) != 16 {
		t.Errorf("Expected 16 neurons in hidden layer, got %d", len(network.Layers()[1].neurons))
	}
	if len(network.Layers()[2].neurons) != 16 {
		t.Errorf("Expected 16 neurons in hidden layer, got %d", len(network.Layers()[1].neurons))
	}
	if len(network.Layers()[3].neurons) != 10 {
		t.Errorf("Expected 10 neuron in output layer, got %d", len(network.Layers()[2].neurons))
	}

	// テスト: コネクション数が正しいか
	expectedLayerConnCounts := []int{0, 16, 16, 10}
	expectedConnCounts := []int{0, 784, 16, 16}

	if len(network.Connections()) != len(expectedConnCounts) {
		t.Errorf("Expected %d connection layers, got %d", len(expectedConnCounts), len(network.Connections()))
	}

	allConnCounts := 0
	for layerIndex, layerConn := range network.Connections() {
		// それぞれのレイヤーに属するニューロンには、前の層のすべてのニューロンから接続があり、ニューロン毎の配列として保持されているが、配列の個数が正しいか検証
		if len(layerConn) != expectedLayerConnCounts[layerIndex] {
			t.Errorf("Expected %d layer connections, got %d", expectedLayerConnCounts[layerIndex], len(layerConn))
		}

		for _, conn := range layerConn {
			// それぞれのニューロンには、前の層のすべてのニューロンから接続があるが、接続の個数が正しいか検証
			if len(conn) != expectedConnCounts[layerIndex] {
				t.Errorf("Expected %d layer connections, got %d", expectedConnCounts[layerIndex], len(conn))
			}
			allConnCounts += len(conn)
		}
	}

	expectedAllConnCounts := 784*16 + 16*16 + 16*10
	if allConnCounts != expectedAllConnCounts {
		t.Errorf("Expected %d layer connections, got %d", expectedAllConnCounts, allConnCounts)
	}

	// テスト: UpdateNeuronConnections が正しく動作するか
	testLayerIndex := 1  // 最初の隠れ層
	testNeuronIndex := 3 // 最初のニューロン

	// 元の重みを保存
	originalWeights := make([]Number, len(network.Connections()[testLayerIndex][testNeuronIndex]))
	for i, conn := range network.Connections()[testLayerIndex][testNeuronIndex] {
		originalWeights[i] = conn.weight
	}

	// テスト用の重み調整値
	weightAdjustments := make([]Number, len(originalWeights))
	for i := range weightAdjustments {
		weightAdjustments[i] = 0.1 // 各重みを0.1増加
	}

	// AdjustNeuronConnections を呼び出し
	err := network.AdjustNeuronConnections(testLayerIndex, testNeuronIndex, weightAdjustments)
	if err != nil {
		t.Errorf("AdjustNeuronConnections returned an error: %v", err)
	}

	// 重みが正しく更新されたか確認
	for i, conn := range network.Connections()[testLayerIndex][testNeuronIndex] {
		expectedWeight := originalWeights[i] + 0.1
		if conn.weight != expectedWeight {
			t.Errorf("Weight not correctly updated. Expected %f, got %f", expectedWeight, conn.weight)
		}
	}

	// エラーケースのテスト
	err = network.AdjustNeuronConnections(-1, 0, weightAdjustments)
	if err == nil {
		t.Error("Expected error for invalid layer index, got nil")
	}

	err = network.AdjustNeuronConnections(testLayerIndex, -1, weightAdjustments)
	if err == nil {
		t.Error("Expected error for invalid neuron index, got nil")
	}

	err = network.AdjustNeuronConnections(testLayerIndex, testNeuronIndex, []Number{0.1}) // 不正な長さの調整値
	if err == nil {
		t.Error("Expected error for mismatched adjustment length, got nil")
	}
}
