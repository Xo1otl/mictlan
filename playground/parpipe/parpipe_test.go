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

// --- 汎用ワーカープール ---
func workerPool[Req, Res any](
	numWorkers int,
	taskFn func(Req) Res,
	reqCh <-chan Req,
	resCh chan<- Res,
	wg *sync.WaitGroup,
) {
	for range numWorkers {
		wg.Go(func() {
			for req := range reqCh {
				resCh <- taskFn(req)
			}
		})
	}
}

// --- 制御ループ ---
// TODO: リアルタイムロギングのための送信チャンネルを追加して、EventをDispatchする
func controlLoop[S, Req, Res any](
	dispatch func(state S, reqCh chan<- Req),
	propagate func(state S, result Res),
	shouldTerminate func(state S) bool,
	reqCh chan<- Req,
	resCh <-chan Res,
	state S,
) {
	fmt.Println("--- Starting Event-Driven Control Loop ---")

	// 1. パイプラインの初期充填
	dispatch(state, reqCh)

	// 2. イベント駆動ループ
	for result := range resCh {
		propagate(state, result)
		if shouldTerminate(state) {
			break
		}
		dispatch(state, reqCh) // 結果を処理した後、再度タスク発行
	}

	// 3. クリーンシャットダウンのメッセージング
	fmt.Println("\n--- All pending tasks finished. ---")
	fmt.Println("--- Control Loop Finished ---")
}

// --- パイプラインを流れるデータ ---
type TaskRequest struct {
	IslandID   int
	Population []Individual
}

type TaskContext struct {
	IslandID  int
	Offspring Individual
}

type TaskResult struct {
	IslandID       int
	EvaluatedChild Individual
}

// --- 状態管理 ---
type State struct {
	Islands            []Island
	GlobalBest         Individual
	PendingIslands     map[int]bool
	EvaluationsCount   int
	AvailableIslandIDs []int // 利用可能なリソースIDのプール
}

// --- データ構造定義 ---
type Individual struct {
	Chromosome []float64
	Fitness    float64
}

func (i Individual) Clone() Individual {
	newChromosome := make([]float64, len(i.Chromosome))
	copy(newChromosome, i.Chromosome)
	return Individual{
		Chromosome: newChromosome,
		Fitness:    i.Fitness,
	}
}

type Island struct {
	ID         int
	Population []Individual
}

func task1(req TaskRequest) TaskContext {
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
		childChromosome = make([]float64, numDimensions)
		copy(childChromosome, parent1.Chromosome)
	}

	for i := range childChromosome {
		if rand.Float64() < mutationRate {
			childChromosome[i] += rand.NormFloat64() * mutationStdDev
			childChromosome[i] = math.Max(searchMin, math.Min(searchMax, childChromosome[i]))
		}
	}
	offspring := Individual{Chromosome: childChromosome, Fitness: math.MaxFloat64}
	return TaskContext{IslandID: req.IslandID, Offspring: offspring}
}

func task2(ctx TaskContext) TaskResult {
	evaluatedChild := ctx.Offspring
	evaluatedChild.Fitness = rastrigin(evaluatedChild.Chromosome)
	return TaskResult{IslandID: ctx.IslandID, EvaluatedChild: evaluatedChild}
}

// タスク発行のロジックを完全にカプセル化する。
// dispatch は発行可能なタスクがなくなるか、終了条件に達するまでタスクをreqChに送信し続けます。
func dispatch(state *State, reqCh chan<- TaskRequest) {
	for {
		// 終了条件（評価回数）またはリソース不足のチェック
		if state.EvaluationsCount >= totalEvaluations || len(state.AvailableIslandIDs) == 0 {
			return // 発行できるタスクがない、または終了条件を満たした
		}

		// 利用可能な島からランダムに選択
		randIndex := rand.Intn(len(state.AvailableIslandIDs))
		islandID := state.AvailableIslandIDs[randIndex]

		// --- State Mutation ---
		// 利用可能プールからIDを削除し、実行中プールへ移動
		state.AvailableIslandIDs = append(state.AvailableIslandIDs[:randIndex], state.AvailableIslandIDs[randIndex+1:]...)
		state.PendingIslands[islandID] = true
		// --- End State Mutation ---

		// タスクを生成してチャネルに送信
		reqCh <- TaskRequest{
			IslandID:   islandID,
			Population: state.Islands[islandID].Population,
		}
	}
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
		state.GlobalBest = evaluatedChild.Clone()
	}

	// 移住のトリガー
	if state.EvaluationsCount%migrationInterval == 0 && state.EvaluationsCount > 0 {
		if state.EvaluationsCount%(migrationInterval*10) == 0 {
			fmt.Printf("Eval: %d / %d, Best Fitness: %.4f\n",
				state.EvaluationsCount, totalEvaluations, state.GlobalBest.Fitness)
		}
		migrate(state.Islands)
	}
	// --- End State Mutation ---
}

// shouldTerminate は、メインの処理が終了し、かつ全てのペンディングタスクが完了したかどうかを判断します。
func shouldTerminate(state *State) bool {
	isEvaluationLimitReached := state.EvaluationsCount >= totalEvaluations
	areAllTasksDone := len(state.PendingIslands) == 0

	return isEvaluationLimitReached && areAllTasksDone
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

func migrate(islands []Island) {
	if len(islands) <= 1 {
		return
	}

	migrantsPerIsland := make([][]Individual, len(islands))
	for i, island := range islands {
		sort.Slice(island.Population, func(a, b int) bool {
			return island.Population[a].Fitness < island.Population[b].Fitness
		})
		migrants := make([]Individual, migrationSize)
		for j := range migrationSize {
			migrants[j] = island.Population[j].Clone()
		}
		migrantsPerIsland[i] = migrants
	}

	for i := range islands {
		targetIslandIndex := (i + 1) % len(islands)
		migrants := migrantsPerIsland[i]
		targetIsland := &islands[targetIslandIndex]

		sort.Slice(targetIsland.Population, func(a, b int) bool {
			return targetIsland.Population[a].Fitness > targetIsland.Population[b].Fitness
		})
		for j := 0; j < migrationSize && j < len(targetIsland.Population); j++ {
			targetIsland.Population[j] = migrants[j]
		}
	}
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
				globalBest = ind.Clone()
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
	searchMin      = -5.12
	searchMax      = 5.12
	mutationRate   = 1.0 / numDimensions
	mutationStdDev = (searchMax - searchMin) * 0.05
)

func TestGA(t *testing.T) {
	task1RequestChannel := make(chan TaskRequest, concurrency)
	task2RequestChannel := make(chan TaskContext, concurrency)
	taskResultChannel := make(chan TaskResult, concurrency)

	var wgTask1, wgTask2 sync.WaitGroup

	workerPool(concurrency, task1, task1RequestChannel, task2RequestChannel, &wgTask1)
	workerPool(concurrency, task2, task2RequestChannel, taskResultChannel, &wgTask2)

	go func() {
		wgTask1.Wait()
		close(task2RequestChannel)
	}()

	initialState := initializeState()
	startTime := time.Now()

	controlLoop(
		dispatch,
		propagate,
		shouldTerminate,
		task1RequestChannel,
		taskResultChannel,
		initialState,
	)

	// controlLoopが終了したため、チャネルをクローズしてワーカーを終了させる
	close(task1RequestChannel)
	wgTask2.Wait()

	duration := time.Since(startTime)
	fmt.Printf("\n--- Search Finished in %s ---\n", duration)
	fmt.Printf("Final Global Best Fitness: %.8f\n", initialState.GlobalBest.Fitness)

	if initialState.GlobalBest.Fitness > 1.0 {
		t.Errorf("Expected best fitness to be close to 0, but got %f", initialState.GlobalBest.Fitness)
	}
}
