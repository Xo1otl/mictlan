package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// State はシステムの現在の状態を表します。
type State struct {
	value int
}

// TaskRequest は task1 ゴルーチンへのリクエストです。
type TaskRequest struct {
	ID    int   // 発行時のデバッグ/ログ目的でIDを保持
	State State // リクエスト発行時点のState
}

// TaskContext は task1 から connectionLoop を経由して task2 へ渡されるデータです。
type TaskContext struct {
	Task1Result int
	ContextData string
}

// TaskResult は task2 から制御ループへ返される最終結果です。
type TaskResult struct {
	Task2Result int
	ContextData string
}

// task1 はStateに依存するI/Oバウンドな処理をシミュレートします。
func task1(req TaskRequest) (res1 int, ctx string) {
	processingTime := time.Duration(50+rand.Intn(100)) * time.Millisecond
	fmt.Printf("[Task 1] ID: %d, Start: Input State.value = %d. (will take %v)\n", req.ID, req.State.value, processingTime)
	time.Sleep(processingTime)
	res1 = req.State.value * 2
	ctx = fmt.Sprintf("context_from_ID_%d", req.ID)
	fmt.Printf("[Task 1] ID: %d, Done: res1 = %d\n", req.ID, res1)
	return res1, ctx
}

// task2 は別のI/Oバウンドな処理をシミュレートします。
func task2(res1 int) (res2 int) {
	processingTime := time.Duration(50+rand.Intn(100)) * time.Millisecond
	fmt.Printf("  [Task 2] Start: res1 = %d. (will take %v)\n", res1, processingTime)
	time.Sleep(processingTime)
	res2 = res1 + 5
	fmt.Printf("  [Task 2] Done: res1 = %d -> res2 = %d\n", res1, res2)
	return res2
}

// reduce は task2 の結果とコンテキストを使い、現在のStateを更新します。
func reduce(result TaskResult, currentState State) State {
	fmt.Printf("    [Reduce] Start: Applying res2 = %d with ctx = '%s' to currentState.value = %d\n", result.Task2Result, result.ContextData, currentState.value)
	newState := State{value: currentState.value + result.Task2Result}
	fmt.Printf("    [Reduce] Done: New State.value = %d\n\n", newState.value)
	return newState
}

// task1WorkerPool は task1 の結果を connectionLoop へ送ります。
func task1WorkerPool(numWorkers int, reqCh <-chan TaskRequest, resCh chan<- TaskContext, wg *sync.WaitGroup) {
	for range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for req := range reqCh {
				res1, ctxData := task1(req)
				resCh <- TaskContext{
					Task1Result: res1,
					ContextData: ctxData,
				}
			}
		}()
	}
}

// task2WorkerPool は connectionLoop からリクエストを受け取ります。
func task2WorkerPool(numWorkers int, reqCh <-chan TaskContext, resCh chan<- TaskResult, wg *sync.WaitGroup) {
	for range numWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx := range reqCh {
				res2 := task2(ctx.Task1Result)
				resCh <- TaskResult{
					Task2Result: res2,
					ContextData: ctx.ContextData,
				}
			}
		}()
	}
}

// ★ DIのために関数型を定義
type AggregateFunc func(in <-chan TaskContext, out chan<- TaskContext)

// passThroughAggregator は受け取ったデータを単純に次のチャネルに横流しします。
func passThroughAggregator(in <-chan TaskContext, out chan<- TaskContext) {
	fmt.Println("    [Conn Loop] Started with PassThroughAggregator.")
	for ctx := range in {
		out <- ctx
	}
	fmt.Println("    [Conn Loop] Finished PassThroughAggregator.")
}

// batchingAggregator はデータをバッチ処理（一定数たまるかタイムアウトでまとめて流す）します。
func batchingAggregator(in <-chan TaskContext, out chan<- TaskContext) {
	const batchSize = 3
	const timeout = 200 * time.Millisecond
	batch := make([]TaskContext, 0, batchSize)

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	fmt.Println("    [Conn Loop] Started with BatchingAggregator.")

	flush := func() {
		fmt.Printf("    [Conn Loop] Flushing batch of %d items.\n", len(batch))
		for _, item := range batch {
			out <- item
		}
		batch = make([]TaskContext, 0, batchSize)
	}

	for {
		select {
		case ctx, ok := <-in:
			if !ok { // 入力チャネルが閉じた
				if len(batch) > 0 {
					flush()
				}
				fmt.Println("    [Conn Loop] Finished BatchingAggregator.")
				return
			}

			batch = append(batch, ctx)
			if len(batch) >= batchSize {
				flush()
				if !timer.Stop() {
					<-timer.C
				}
				timer.Reset(timeout)
			}
		case <-timer.C:
			if len(batch) > 0 {
				flush()
			}
			timer.Reset(timeout)
		}
	}
}

// connectionWorker はDIされたAggregatorを使ってtask1とtask2を繋ぎます。
func connectionWorker(in <-chan TaskContext, out chan<- TaskContext, wg *sync.WaitGroup, aggregator AggregateFunc) {
	defer wg.Done()
	defer close(out) // このワーカーが終了したら、出力チャネルを閉じるのが責務
	aggregator(in, out)
}

// controlLoop はシステムのメインロジックを実行します。
// ★ エラーの原因だったゴルーチンを削除し、チャネルを閉じる責務をループの最後に移動
func controlLoop(reqCh chan<- TaskRequest, resCh <-chan TaskResult, totalTasks int, concurrency int, initialState State) State {
	fmt.Println("--- Starting Control Loop ---")
	currentState := initialState
	tasksIssued := 0
	tasksCompleted := 0

	// 1. 初期化: 並行数分のタスクを最初に発行
	fmt.Printf("--- Issuing initial %d tasks ---\n", concurrency)
	for i := 0; i < concurrency && i < totalTasks; i++ {
		req := TaskRequest{ID: tasksIssued, State: currentState}
		fmt.Printf("[Controller] Issuing initial task ID %d\n", req.ID)
		reqCh <- req
		tasksIssued++
	}

	// 2. メインループ: 1つ完了したら、1つ発行する
	for tasksCompleted < totalTasks {
		result := <-resCh
		tasksCompleted++
		fmt.Printf("[Controller] Received a result. (Completed: %d/%d)\n", tasksCompleted, totalTasks)

		currentState = reduce(result, currentState)

		if tasksIssued < totalTasks {
			req := TaskRequest{ID: tasksIssued, State: currentState}
			fmt.Printf("[Controller] Issuing next task ID %d with new State.value = %d\n", req.ID, req.State.value)
			reqCh <- req
			tasksIssued++
		}
	}

	// 3. 終了処理: 全てのタスクが完了したので、これ以上発行しないことをワーカーに伝える
	close(reqCh)

	fmt.Println("--- Control Loop Finished ---")
	return currentState
}

// runSimulation はDIされたaggregatorを使って一連のシミュレーションを実行します。
func runSimulation(totalTasks, concurrency int, initialState State, aggregator AggregateFunc) {
	// ---- チャネルの準備 ----
	task1RequestChannel := make(chan TaskRequest, concurrency)
	task1ResultChannel := make(chan TaskContext, concurrency)  // task1 -> connectionLoop
	task2RequestChannel := make(chan TaskContext, concurrency) // connectionLoop -> task2
	taskResultChannel := make(chan TaskResult, concurrency)    // task2 -> controlLoop

	// ---- ワーカーと接続ループの起動 ----
	var wg1, wg2, wgConn sync.WaitGroup

	// Stage 1: task1 ワーカープール
	task1WorkerPool(concurrency, task1RequestChannel, task1ResultChannel, &wg1)

	// Stage 2: 接続ワーカー (DIされたaggregatorを使用)
	wgConn.Add(1)
	go connectionWorker(task1ResultChannel, task2RequestChannel, &wgConn, aggregator)

	// Stage 3: task2 ワーカープール
	task2WorkerPool(concurrency, task2RequestChannel, taskResultChannel, &wg2)

	// ---- パイプラインの連携 (チャネルを適切に閉じるためのゴルーチン) ----
	go func() {
		wg1.Wait()
		close(task1ResultChannel) // task1ワーカーが全て終了したら、connectionLoopへの入力チャネルを閉じる
	}()

	// ---- 制御ループの実行 ----
	finalState := controlLoop(task1RequestChannel, taskResultChannel, totalTasks, concurrency, initialState)

	// ---- 最終結果の表示と終了処理 ----
	fmt.Printf("\nFinal State: value = %d\n", finalState.value)

	// 全てのゴルーチンが正常に終了するのを待つ
	wgConn.Wait()
	wg2.Wait()
	fmt.Println("All workers finished. Exiting.")
}

func main() {
	// ---- 定数と初期状態 ----
	totalTasks := 10
	concurrency := 3
	initialState := State{value: 1000}

	// --- シミュレーション1: Aggregatorなし (単純なパススルー) ---
	fmt.Println("======================================================")
	fmt.Println("=== Running Simulation with PassThrough Aggregator ===")
	fmt.Println("======================================================")
	runSimulation(totalTasks, concurrency, initialState, passThroughAggregator)

	// --- シミュレーション2: バッチ処理Aggregatorあり ---
	fmt.Println("=================================================")
	fmt.Println("=== Running Simulation with Batching Aggregator ===")
	fmt.Println("=================================================")
	runSimulation(totalTasks, concurrency, initialState, batchingAggregator)
}
