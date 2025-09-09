package main_test

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
	"time"
)

// --- GA/Rastrigin関数関連の定数 ---
const (
	// -- GAパラメータ (Python版の高性能な設定を反映) --
	numIslands     = 5   // 島の数
	populationSize = 50  // 各島の個体数
	numDimensions  = 30  // ラスタリギン関数の次元数
	crossoverRate  = 0.9 // 交叉率
	blxAlpha       = 0.5 // BLX-α交叉のパラメータ
	tournamentSize = 5   // トーナメント選択のサイズ

	// -- 実行制御パラメータ --
	totalEvaluations  = 250000 // 総評価回数 (次元数増加に伴い調整)
	concurrency       = 8      // 並列実行数 (CPUコア数などに合わせる)
	migrationInterval = 25     // この評価回数ごとに移住を行う
	migrationSize     = 5      // 一度の移住で交換される個体数
)

var (
	// 探索範囲と、それに依存する突然変異パラメータ
	searchMin      = -5.12
	searchMax      = 5.12
	mutationRate   = 1.0 / numDimensions            // 突然変異率 (次元数に依存)
	mutationStdDev = (searchMax - searchMin) * 0.05 // 突然変異の標準偏差 (探索範囲に依存)
)

// --- データ構造定義 ---

// Individual はGAの一個体を表します。
type Individual struct {
	Chromosome []float64 // 遺伝子情報 (関数の入力ベクトル)
	Fitness    float64   // 適応度 (関数の評価値、小さいほど良い)
}

// Island は個体の集合である島を表します。
type Island struct {
	ID         int
	Population []Individual
}

// SearchState は探索全体の状態を保持します。
type SearchState struct {
	Islands    []Island
	GlobalBest Individual
}

// ControlState は非同期処理の制御状態を保持します。
type ControlState struct {
	PendingIslands   map[int]bool
	EvaluationsCount int
}

// TaskRequest は task1 (子個体生成) ゴルーチンへのリクエストです。
type TaskRequest struct {
	RequestID  int
	IslandID   int
	Population []Individual
}

// TaskContext は task1 から task2 へ渡されるデータです。
type TaskContext struct {
	RequestID int
	IslandID  int
	Offspring Individual
}

// TaskResult は task2 (評価) から制御ループへ返される最終結果です。
type TaskResult struct {
	RequestID      int
	IslandID       int
	EvaluatedChild Individual
}

// --- ユーティリティ関数 ---

func newIndividual() Individual {
	chromosome := make([]float64, numDimensions)
	for i := range chromosome {
		chromosome[i] = searchMin + rand.Float64()*(searchMax-searchMin)
	}
	return Individual{Chromosome: chromosome, Fitness: math.MaxFloat64}
}

func rastrigin(chromosome []float64) float64 {
	a := 10.0
	sum := a * float64(len(chromosome))
	for _, x := range chromosome {
		sum += x*x - a*math.Cos(2*math.Pi*x)
	}
	return sum
}

// --- コアロジック: 交叉、突然変異、選択など ---

// crossoverBLXAlpha は実数値GAで高性能な交叉手法です。
func crossoverBLXAlpha(p1, p2 []float64, alpha float64) []float64 {
	child := make([]float64, len(p1))
	for i := range p1 {
		d := math.Abs(p1[i] - p2[i])
		minGene := math.Min(p1[i], p2[i]) - alpha*d
		maxGene := math.Max(p1[i], p2[i]) + alpha*d

		minGene = math.Max(searchMin, minGene)
		maxGene = math.Min(searchMax, maxGene)

		// minGene > maxGeneの場合を考慮
		if minGene > maxGene {
			minGene, maxGene = maxGene, minGene
		}

		child[i] = minGene + rand.Float64()*(maxGene-minGene)
	}
	return child
}

// task1 は、親を選択し、交叉・突然変異によって子を生成します。
func task1(req TaskRequest) TaskContext {
	// 1. トーナメント選択で親を2体選ぶ
	tournament := func() Individual {
		best := req.Population[rand.Intn(len(req.Population))]
		for i := 1; i < tournamentSize; i++ {
			competitor := req.Population[rand.Intn(len(req.Population))]
			if competitor.Fitness < best.Fitness {
				best = competitor
			}
		}
		return best
	}
	parent1 := tournament()
	parent2 := tournament()

	// 2. 交叉 (高性能なBLX-α交叉を使用)
	var childChromosome []float64
	if rand.Float64() < crossoverRate {
		childChromosome = crossoverBLXAlpha(parent1.Chromosome, parent2.Chromosome, blxAlpha)
	} else {
		childChromosome = make([]float64, numDimensions)
		copy(childChromosome, parent1.Chromosome)
	}

	// 3. 突然変異 (ガウス分布に従う)
	for i := range childChromosome {
		if rand.Float64() < mutationRate {
			childChromosome[i] += rand.NormFloat64() * mutationStdDev
			childChromosome[i] = math.Max(searchMin, math.Min(searchMax, childChromosome[i]))
		}
	}

	offspring := Individual{Chromosome: childChromosome, Fitness: math.MaxFloat64}
	return TaskContext{
		RequestID: req.RequestID,
		IslandID:  req.IslandID,
		Offspring: offspring,
	}
}

// task2 は子の適応度を評価関数で計算します。
func task2(ctx TaskContext) TaskResult {
	evaluatedChild := ctx.Offspring
	evaluatedChild.Fitness = rastrigin(evaluatedChild.Chromosome)
	return TaskResult{
		RequestID:      ctx.RequestID,
		IslandID:       ctx.IslandID,
		EvaluatedChild: evaluatedChild,
	}
}

// reduce は評価済みの子個体を元の島の個体群に組み込みます（定常状態モデル）。
func reduce(result TaskResult, currentState *SearchState) {
	island := &currentState.Islands[result.IslandID]
	worstIndex := 0
	for i := 1; i < len(island.Population); i++ {
		if island.Population[i].Fitness > island.Population[worstIndex].Fitness {
			worstIndex = i
		}
	}
	if result.EvaluatedChild.Fitness < island.Population[worstIndex].Fitness {
		island.Population[worstIndex] = result.EvaluatedChild
	}
	if result.EvaluatedChild.Fitness < currentState.GlobalBest.Fitness {
		newBestChromosome := make([]float64, len(result.EvaluatedChild.Chromosome))
		copy(newBestChromosome, result.EvaluatedChild.Chromosome)
		currentState.GlobalBest = Individual{
			Chromosome: newBestChromosome,
			Fitness:    result.EvaluatedChild.Fitness,
		}
	}
}

// migrate は島の間で最良個体を複数交換し、多様性を維持します。
func migrate(currentState *SearchState) {
	numIslands := len(currentState.Islands)
	if numIslands <= 1 {
		return
	}
	fmt.Println("--- Performing Migration ---")

	// 1. 各島から移住者（エリート）を選出
	migrantsPerIsland := make([][]Individual, numIslands)
	for i, island := range currentState.Islands {
		popCopy := make([]Individual, len(island.Population))
		copy(popCopy, island.Population)
		sort.Slice(popCopy, func(a, b int) bool {
			return popCopy[a].Fitness < popCopy[b].Fitness
		})
		migrants := make([]Individual, migrationSize)
		for j := 0; j < migrationSize; j++ {
			// 参照ではなく値のコピーを確実に作成
			newChromosome := make([]float64, numDimensions)
			copy(newChromosome, popCopy[j].Chromosome)
			migrants[j] = Individual{Chromosome: newChromosome, Fitness: popCopy[j].Fitness}
		}
		migrantsPerIsland[i] = migrants
	}

	// 2. リング状に移住を実行 (i番目の島のエリートがi+1番目の島へ)
	for i := 0; i < numIslands; i++ {
		sourceIslandIndex := i
		targetIslandIndex := (i + 1) % numIslands
		migrants := migrantsPerIsland[sourceIslandIndex]
		targetIsland := &currentState.Islands[targetIslandIndex]

		// 移住先の最悪個体群を移住者で置き換え
		sort.Slice(targetIsland.Population, func(a, b int) bool {
			return targetIsland.Population[a].Fitness > targetIsland.Population[b].Fitness
		})
		for j := 0; j < migrationSize; j++ {
			if j < len(targetIsland.Population) {
				targetIsland.Population[j] = migrants[j]
			}
		}
	}
}

// --- ワーカープールとパイプライン ---

func task1WorkerPool(numWorkers int, reqCh <-chan TaskRequest, resCh chan<- TaskContext, wg *sync.WaitGroup) {
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for req := range reqCh {
				resCh <- task1(req)
			}
		}()
	}
}

func task2WorkerPool(numWorkers int, reqCh <-chan TaskContext, resCh chan<- TaskResult, wg *sync.WaitGroup) {
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx := range reqCh {
				resCh <- task2(ctx)
			}
		}()
	}
}

// --- 制御ループ ---

func controlLoop(reqCh chan<- TaskRequest, resCh <-chan TaskResult, searchState *SearchState) {
	fmt.Println("--- Starting Control Loop ---")

	controlState := ControlState{
		PendingIslands:   make(map[int]bool),
		EvaluationsCount: 0,
	}
	requestIDCounter := 0
	lastMigrationCount := 0

	for controlState.EvaluationsCount < totalEvaluations {
		canIssue := len(controlState.PendingIslands) < concurrency
		if canIssue {
			targetIslandID := -1
			for id := 0; id < numIslands; id++ {
				if !controlState.PendingIslands[id] {
					targetIslandID = id
					break
				}
			}
			if targetIslandID != -1 {
				controlState.PendingIslands[targetIslandID] = true
				req := TaskRequest{
					RequestID:  requestIDCounter,
					IslandID:   targetIslandID,
					Population: searchState.Islands[targetIslandID].Population,
				}
				reqCh <- req
				requestIDCounter++
				continue
			}
		}

		result := <-resCh
		controlState.EvaluationsCount++
		delete(controlState.PendingIslands, result.IslandID)
		reduce(result, searchState)

		if controlState.EvaluationsCount-lastMigrationCount >= migrationInterval {
			fmt.Printf("Evaluations: %d / %d, Current Best Fitness: %.4f\n",
				controlState.EvaluationsCount, totalEvaluations, searchState.GlobalBest.Fitness)
			migrate(searchState)
			lastMigrationCount = controlState.EvaluationsCount
		}
	}

	fmt.Println("--- Evaluation limit reached. Waiting for pending tasks... ---")
	for len(controlState.PendingIslands) > 0 {
		result := <-resCh
		delete(controlState.PendingIslands, result.IslandID)
		reduce(result, searchState)
	}

	close(reqCh)
	fmt.Println("--- Control Loop Finished ---")
}

// --- 初期化と実行 ---

func initializeSearchState() *SearchState {
	islands := make([]Island, numIslands)
	globalBest := Individual{Fitness: math.MaxFloat64}
	initialEvaluationCount := 0
	for i := 0; i < numIslands; i++ {
		population := make([]Individual, populationSize)
		for j := 0; j < populationSize; j++ {
			ind := newIndividual()
			ind.Fitness = rastrigin(ind.Chromosome)
			initialEvaluationCount++
			population[j] = ind
			if ind.Fitness < globalBest.Fitness {
				globalBest = ind
			}
		}
		islands[i] = Island{ID: i, Population: population}
	}
	fmt.Printf("Initialization complete. Evaluated %d individuals.\n", initialEvaluationCount)
	fmt.Printf("Initial best fitness: %.4f\n", globalBest.Fitness)
	return &SearchState{Islands: islands, GlobalBest: globalBest}
}

func TestGeneticAlgorithmExecution(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	task1RequestChannel := make(chan TaskRequest, concurrency)
	task1ToTask2Channel := make(chan TaskContext, concurrency)
	taskResultChannel := make(chan TaskResult, concurrency)

	var wgTask1, wgTask2 sync.WaitGroup
	task1WorkerPool(concurrency, task1RequestChannel, task1ToTask2Channel, &wgTask1)
	task2WorkerPool(concurrency, task1ToTask2Channel, taskResultChannel, &wgTask2)

	initialState := initializeSearchState()
	startTime := time.Now()
	controlLoop(task1RequestChannel, taskResultChannel, initialState)

	wgTask1.Wait()
	close(task1ToTask2Channel)
	wgTask2.Wait()

	duration := time.Since(startTime)
	fmt.Printf("\n--- Search Finished in %s ---\n", duration)
	fmt.Printf("Final Global Best Fitness: %.8f\n", initialState.GlobalBest.Fitness)
}
