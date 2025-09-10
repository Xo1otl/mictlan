# 導入
LLMによる提案と決定論的評価の組み合わせでヒューリスティックな仮説検証を行うagentic algorithm

* 非同期島モデルGAにおけるMutation OperatorでのLLM導入
* Parallel MCTSにおける展開時のLLMの使用

提案、評価それぞれがI/Oバウンドとなりうる(以下ではCPUバウンドでないと仮定する)

島単位の並列化ではなく、stateの構造を工夫して論理的に島モデルとすべきである

このようなアルゴリズムはすべて、別々に並列化すべき二つのタスクがあって、結果により状態が更新される制御ループがあるプログラムとして定式化可能である

# 概要
* 制御ループ (一元的にstateを操作、mutexが不要に)
* task1ループ (提案にあたる)
* task2ループ (評価にあたる)

他に必要そうなもの
* task1に直接stateを投げるのではなく、issue the request
    * okならばchanにreqを投げる
* termination strategy
* propagateでstateの伝播をシンプルな関数にカプセル化

# PoC
```go
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
	// -- GAパラメータ --
	numIslands     = 5   // 島の数
	populationSize = 50  // 各島の個体数
	numDimensions  = 30  // ラスタリギン関数の次元数
	crossoverRate  = 0.9 // 交叉率
	blxAlpha       = 0.5 // BLX-α交叉のパラメータ
	tournamentSize = 5   // トーナメント選択のサイズ

	// -- 実行制御パラメータ --
	totalEvaluations  = 250000 // 総評価回数
	concurrency       = 8      // 並列実行数
	migrationInterval = 25     // この評価回数ごとに移住を行う
	migrationSize     = 5      // 一度の移住で交換される個体数
)

var (
	// 探索範囲と、それに依存する突然変異パラメータ
	searchMin      = -5.12
	searchMax      = 5.12
	mutationRate   = 1.0 / numDimensions
	mutationStdDev = (searchMax - searchMin) * 0.05
)

// --- データ構造定義 ---

// Individual はGAの一個体を表します。
type Individual struct {
	Chromosome []float64
	Fitness    float64
}

// Island は個体の集合である部分個体群（島）を表します。
type Island struct {
	ID         int
	Population []Individual
}

// --- パイプラインを流れるデータ ---

// TaskRequest は task1 (子個体生成) ゴルーチンへのリクエストです。
type TaskRequest struct {
	IslandID   int
	Population []Individual
}

// TaskContext は task1 から task2 へ渡されるデータです。
type TaskContext struct {
	IslandID  int
	Offspring Individual
}

// TaskResult は task2 (評価) から制御ループへ返される最終結果です。
type TaskResult struct {
	IslandID       int
	EvaluatedChild Individual
}

// --- 状態管理 ---

// State はアプリケーションの全ての状態を保持する単一の構造体です
type State struct {
	Islands            []Island
	GlobalBest         Individual
	PendingIslands     map[int]bool
	EvaluationsCount   int
	AvailableIslandIDs []int // 利用可能なリソースIDのプール
}

// issue は利用可能な島を選択し、タスクを発行して、状態を更新します。
func issue(state *State) (TaskRequest, bool) {
	if len(state.AvailableIslandIDs) == 0 {
		return TaskRequest{}, false
	}
	// 利用可能な島からランダムに選択
	randIndex := rand.Intn(len(state.AvailableIslandIDs))
	islandID := state.AvailableIslandIDs[randIndex]

	// --- State Mutation ---
	// 利用可能プールからIDを削除し、実行中プールへ移動
	state.AvailableIslandIDs = append(state.AvailableIslandIDs[:randIndex], state.AvailableIslandIDs[randIndex+1:]...)
	state.PendingIslands[islandID] = true
	// --- End State Mutation ---

	return TaskRequest{
		IslandID:   islandID,
		Population: state.Islands[islandID].Population,
	}, true
}

// propagate は完了したタスクの結果をStateに反映させます。
func propagate(state *State, result TaskResult) {
	islandID := result.IslandID
	evaluatedChild := result.EvaluatedChild

	// --- State Mutation ---
	// Control Stateの更新
	delete(state.PendingIslands, islandID)
	state.EvaluationsCount++
	state.AvailableIslandIDs = append(state.AvailableIslandIDs, islandID)

	// Search Stateの更新 (定常状態モデル: 最悪個体を置き換え)
	island := &state.Islands[islandID]
	worstIndex := 0
	for i := 1; i < len(island.Population); i++ {
		if island.Population[i].Fitness > island.Population[worstIndex].Fitness {
			worstIndex = i
		}
	}
	if evaluatedChild.Fitness < island.Population[worstIndex].Fitness {
		island.Population[worstIndex] = evaluatedChild
	}

	// グローバルベストの更新
	if evaluatedChild.Fitness < state.GlobalBest.Fitness {
		newBestChromosome := make([]float64, len(evaluatedChild.Chromosome))
		copy(newBestChromosome, evaluatedChild.Chromosome)
		state.GlobalBest = Individual{
			Chromosome: newBestChromosome,
			Fitness:    evaluatedChild.Fitness,
		}
	}

	// 移住のトリガー
	if state.EvaluationsCount%migrationInterval == 0 && state.EvaluationsCount > 0 {
		// ログ出力を少し整理
		if state.EvaluationsCount%(migrationInterval*10) == 0 {
			fmt.Printf("Eval: %d / %d, Best Fitness: %.4f\n",
				state.EvaluationsCount, totalEvaluations, state.GlobalBest.Fitness)
		}
		migrate(state.Islands)
	}
	// --- End State Mutation ---
}

// --- 汎用コンポーネントの型定義 (DI用) ---

// ShouldContinue は、現在の状態を元にループを継続するかどうかを判断する純粋関数です
type ShouldContinue func(state *State) bool

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

func crossoverBLXAlpha(p1, p2 []float64, alpha float64) []float64 {
	child := make([]float64, len(p1))
	for i := range p1 {
		d := math.Abs(p1[i] - p2[i])
		minGene := math.Min(p1[i], p2[i]) - alpha*d
		maxGene := math.Max(p1[i], p2[i]) + alpha*d
		minGene = math.Max(searchMin, minGene)
		maxGene = math.Min(searchMax, maxGene)
		if minGene > maxGene {
			minGene, maxGene = maxGene, minGene
		}
		child[i] = minGene + rand.Float64()*(maxGene-minGene)
	}
	return child
}

// task1 は子個体を生成します (交叉と突然変異)
func task1(req TaskRequest) TaskContext {
	// トーナメント選択
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
	parent1, parent2 := tournament(), tournament()

	var childChromosome []float64
	if rand.Float64() < crossoverRate {
		childChromosome = crossoverBLXAlpha(parent1.Chromosome, parent2.Chromosome, blxAlpha)
	} else {
		// 交叉しない場合は親1をコピー
		childChromosome = make([]float64, numDimensions)
		copy(childChromosome, parent1.Chromosome)
	}

	// 突然変異
	for i := range childChromosome {
		if rand.Float64() < mutationRate {
			childChromosome[i] += rand.NormFloat64() * mutationStdDev
			// 範囲内に収める
			childChromosome[i] = math.Max(searchMin, math.Min(searchMax, childChromosome[i]))
		}
	}
	offspring := Individual{Chromosome: childChromosome, Fitness: math.MaxFloat64}
	return TaskContext{IslandID: req.IslandID, Offspring: offspring}
}

// task2 は個体のフィットネスを評価します
func task2(ctx TaskContext) TaskResult {
	evaluatedChild := ctx.Offspring
	evaluatedChild.Fitness = rastrigin(evaluatedChild.Chromosome)
	return TaskResult{IslandID: ctx.IslandID, EvaluatedChild: evaluatedChild}
}

// migrate は島の個体を交換し、多様性を維持します。
func migrate(islands []Island) {
	if len(islands) <= 1 {
		return
	}

	migrantsPerIsland := make([][]Individual, len(islands))
	for i, island := range islands {
		// 最良個体を選択するためにソート
		sort.Slice(island.Population, func(a, b int) bool {
			return island.Population[a].Fitness < island.Population[b].Fitness
		})
		migrants := make([]Individual, migrationSize)
		for j := 0; j < migrationSize; j++ {
			// 安全なコピーを作成
			newChromosome := make([]float64, numDimensions)
			copy(newChromosome, island.Population[j].Chromosome)
			migrants[j] = Individual{Chromosome: newChromosome, Fitness: island.Population[j].Fitness}
		}
		migrantsPerIsland[i] = migrants
	}

	// リング状に移住を実行 (i番目の島から i+1 番目の島へ)
	for i := range islands {
		targetIslandIndex := (i + 1) % len(islands)
		migrants := migrantsPerIsland[i]
		targetIsland := &islands[targetIslandIndex]

		// 最悪個体を置き換えるために逆順ソート
		sort.Slice(targetIsland.Population, func(a, b int) bool {
			return targetIsland.Population[a].Fitness > targetIsland.Population[b].Fitness
		})
		for j := 0; j < migrationSize && j < len(targetIsland.Population); j++ {
			targetIsland.Population[j] = migrants[j]
		}
	}
}

// --- ワーカープールとパイプライン ---

func task1WorkerPool(numWorkers int, reqCh <-chan TaskRequest, resCh chan<- TaskContext, wg *sync.WaitGroup) {
	for range numWorkers {
		go func() {
			defer wg.Done()
			for req := range reqCh {
				resCh <- task1(req)
			}
		}()
	}
}

func task2WorkerPool(numWorkers int, reqCh <-chan TaskContext, resCh chan<- TaskResult, wg *sync.WaitGroup) {
	for range numWorkers {
		go func() {
			defer wg.Done()
			for ctx := range reqCh {
				resCh <- task2(ctx)
			}
		}()
	}
}

// --- 制御ループ ---

// controlLoop はドメイン知識を持たない、汎用的な並行処理オーケストレーターです。
func controlLoop(
	reqCh chan<- TaskRequest,
	resCh <-chan TaskResult,
	state *State,
	shouldContinue ShouldContinue,
) {
	fmt.Println("--- Starting Control Loop ---")

	// 初期タスクの投入 (パイプラインを埋める)
	for range concurrency {
		if !shouldContinue(state) {
			break
		}
		if req, ok := issue(state); ok {
			reqCh <- req
		}
	}

	for shouldContinue(state) {
		result := <-resCh
		propagate(state, result)

		if shouldContinue(state) {
			if req, ok := issue(state); ok {
				reqCh <- req
			}
		}
	}

	fmt.Println("\n--- Evaluation limit reached. Waiting for pending tasks... ---")
	for len(state.PendingIslands) > 0 {
		result := <-resCh
		propagate(state, result)
	}

	// 全タスク投入後、リクエストチャネルを閉じる
	close(reqCh)
	fmt.Println("--- Control Loop Finished ---")
}

// --- 初期化と実行 ---
func initializeState() *State {
	islands := make([]Island, numIslands)
	globalBest := Individual{Fitness: math.MaxFloat64}
	initialEvaluationCount := 0
	availableIDs := make([]int, numIslands)

	for i := range numIslands {
		availableIDs[i] = i
		population := make([]Individual, populationSize)
		for j := range populationSize {
			ind := newIndividual()
			ind.Fitness = rastrigin(ind.Chromosome)
			initialEvaluationCount++
			population[j] = ind
			if ind.Fitness < globalBest.Fitness {
				// 初期個体群の中からベストな個体をディープコピーして保持
				newBestChromosome := make([]float64, len(ind.Chromosome))
				copy(newBestChromosome, ind.Chromosome)
				globalBest = Individual{
					Chromosome: newBestChromosome,
					Fitness:    ind.Fitness,
				}
			}
		}
		islands[i] = Island{ID: i, Population: population}
	}
	fmt.Printf("Initialization complete. Evaluated %d individuals.\n", initialEvaluationCount)
	fmt.Printf("Initial best fitness: %.4f\n", globalBest.Fitness)

	return &State{
		Islands:            islands,
		GlobalBest:         globalBest,
		PendingIslands:     make(map[int]bool),
		EvaluationsCount:   initialEvaluationCount,
		AvailableIslandIDs: availableIDs,
	}
}

// --- DIされる具象関数の実装 ---

// evaluationCountCondition は、評価回数に基づくTerminationConditionを生成するファクトリ関数です。
func evaluationCountCondition(totalEvals int) ShouldContinue {
	return func(state *State) bool {
		return state.EvaluationsCount < totalEvals
	}
}

func TestGA(t *testing.T) {
	task1RequestChannel := make(chan TaskRequest, concurrency)
	task2RequestChannel := make(chan TaskContext, concurrency)
	taskResultChannel := make(chan TaskResult, concurrency)

	var wgTask1, wgTask2 sync.WaitGroup
	wgTask1.Add(concurrency)
	wgTask2.Add(concurrency)

	// ワーカープールを起動
	task1WorkerPool(concurrency, task1RequestChannel, task2RequestChannel, &wgTask1)
	task2WorkerPool(concurrency, task2RequestChannel, taskResultChannel, &wgTask2)

	// task1の全ワーカーが終了したら、task2の入力チャネルを閉じる
	go func() {
		wgTask1.Wait()
		close(task2RequestChannel)
	}()

	initialState := initializeState()
	startTime := time.Now()

	// 依存性を注入して実行
	controlLoop(
		task1RequestChannel,
		taskResultChannel,
		initialState,
		evaluationCountCondition(totalEvaluations),
	)

	// task2のワーカーが全て終了するのを待つ
	wgTask2.Wait()

	duration := time.Since(startTime)
	fmt.Printf("\n--- Search Finished in %s ---\n", duration)
	fmt.Printf("Final Global Best Fitness: %.8f\n", initialState.GlobalBest.Fitness)

	// Rastrigin関数の最適解は0なので、それに近い値が出ていることを確認
	if initialState.GlobalBest.Fitness > 1.0 {
		t.Errorf("Expected best fitness to be close to 0, but got %f", initialState.GlobalBest.Fitness)
	}
}
```

# Questions
* task2のresのみをsubscribeしてtask1の結果を直接task2にパイプしている場合、task1だけが終了してtask2で手間取っている場合に、制御ループが次のtask1の実行を指示することができないのでは？

* shouldContinueがいろんなところで呼ばれている、もっとシンプルなフローにできないか？
* concurrencyの数を制御ループが知っているのはおかしくないか？stateのみによって判断されるべきであり、channel(queue)で分断されている先のworker数について知らなければならない現在の書きかたは根本的な間違いがあると思う

# Your Task
Please answer questions
