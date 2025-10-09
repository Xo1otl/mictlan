package llmsr

import (
	"mictlan/orchestrator/internal/bilevel"
	"mictlan/orchestrator/internal/pb"
	"bufio"
	"context"
	"math/rand"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	maxEvaluations     = 1000
	numIslands         = 2
	eliminationRate    = 0.5
	migrationInterval  = 25
	proposeConcurrency = 1
	observeConcurrency = 1
	testTimeout        = 6 * time.Second
	scoreQuantization  = 2
	t0                 = 0
	n                  = 1
	tp                 = 1.0
)

func useTestRng() *rand.Rand {
	return rand.New(rand.NewSource(42))
}

func newInitialState(t *testing.T, observeFn bilevel.ObserveFunc[ObserveRequest, ObserveResult]) (*DeterministicState, float64) {
	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	initialSkeleton := "-100"
	initialScore := observeFn(ctx, ObserveRequest{Query: Skeleton(initialSkeleton)}).Evidence
	state, err := NewDeterministicState(initialSkeleton, initialScore, maxEvaluations, numIslands, migrationInterval, scoreQuantization, eliminationRate, t0, n, tp, useTestRng())
	if err != nil {
		t.Fatalf("Failed to create initial state: %v", err)
	}
	return state, initialScore
}

func TestLLMSR_WithMock(t *testing.T) {
	runnerState, initialScore := newInitialState(t, MockObserve)
	esState, trace := bilevel.WithEventSourcing(runnerState)
	runLLMSR(t, esState, MockPropose, MockObserve)

	events := trace()
	logStateSummary(t, runnerState, initialScore, events)

	assert.True(t, runnerState.EvaluationsCount >= maxEvaluations, "Should have completed at least the specified number of evaluations")
	assert.Greater(t, getBestScore(runnerState), initialScore, "The final best score should be better (greater) than the initial score")

	t.Log("--- Running Simulation with sequence and Mock workers ---")

	simulatedState, _ := newInitialState(t, MockObserve)
	bilevel.Replay(simulatedState, events)
	logStateSummary(t, simulatedState, initialScore, events)
}

func TestLLMSR_WithGRPCServer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatalf("Failed to get caller information")
	}
	dir := filepath.Dir(filename)
	pythonPath := filepath.Join(dir, "../../../../", ".venv/bin/python")

	cmd := exec.CommandContext(ctx,
		pythonPath, "-u",
		"-c", "import llmsr_worker; llmsr_worker.main()",
	)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("Failed to get stdout pipe: %v", err)
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		t.Fatalf("Failed to start gRPC server: %v", err)
	}
	defer func() {
		if err := cmd.Process.Kill(); err != nil {
			t.Logf("Failed to kill process: %v", err)
		}
		cmd.Wait()
	}()

	serverReady := make(chan bool)
	expectedOutput := "gRPC server started"
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			t.Logf("[gRPC Server]: %s", line)
			if strings.Contains(line, expectedOutput) {
				t.Log("gRPC server is ready.")
				close(serverReady)
				return
			}
		}
	}()
	select {
	case <-serverReady:
	case <-ctx.Done():
		t.Fatal("Timeout waiting for gRPC server to start.")
	}

	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to connect to gRPC server: %v", err)
	}
	defer conn.Close()

	client := pb.NewLLMSRClient(conn)
	proposeFn := NewGRPCPropose(client)
	observeFn := NewGRPCObserve(client)

	state, initialScore := newInitialState(t, observeFn)
	esState, trace := bilevel.WithEventSourcing(state)
	runLLMSR(t, esState, proposeFn, observeFn)

	events := trace()
	logStateSummary(t, state, initialScore, events)

	assert.True(t, state.EvaluationsCount >= maxEvaluations, "Should have completed at least the specified number of evaluations")
	assert.Greater(t, getBestScore(state), initialScore, "The final best score should be better (greater) than the initial score")

	t.Log("--- Running Simulation with gRPC sequence and Mock workers ---")

	simulatedState, _ := newInitialState(t, MockObserve)
	bilevel.Replay(simulatedState, events)
	logStateSummary(t, simulatedState, initialScore, events)
}

func runLLMSR(t *testing.T, state bilevel.State[ProposeRequest, ObserveResult], proposeFn bilevel.ProposeFunc[ProposeRequest, ProposeResult], observeFn bilevel.ObserveFunc[ObserveRequest, ObserveResult]) {
	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	adapter := NewAdapter()

	orchestrator := bilevel.NewOrchestrator(
		proposeFn,
		observeFn,
		proposeConcurrency,
		observeConcurrency,
	)

	errCh := make(chan error, 1)
	go func() {
		err, ok := <-errCh
		if ok {
			t.Logf("Test context canceled by error: %v", err)
			cancel()
		}
	}()

	bilevel.RunWithAdapter(orchestrator, ctx, state, adapter, errCh)

	if ctx.Err() == context.DeadlineExceeded {
		t.Fatal("Test timed out, indicating a potential deadlock or server issue.")
	}
}

func logStateSummary(t *testing.T, state *DeterministicState, initialScore float64, trace []bilevel.StateEvent[ObserveResult]) {
	t.Log("--- State Summary ---")
	t.Logf("Total Islands: %d", len(state.Islands))

	// Sort islands by ID for consistent logging
	sortedIslands := make([]*Island, 0, len(state.Islands))
	for _, island := range state.Islands {
		sortedIslands = append(sortedIslands, island)
	}
	sort.Slice(sortedIslands, func(i, j int) bool {
		return sortedIslands[i].ID < sortedIslands[j].ID
	})

	totalProposeWeightedSum := 0.0

	for _, island := range sortedIslands {
		totalPrograms := 0
		totalScore := 0.0
		for _, cluster := range island.Clusters {
			numPrograms := len(cluster.Programs)
			totalPrograms += numPrograms
			totalScore += cluster.Score * float64(numPrograms)
			totalProposeWeightedSum += float64(numPrograms) * (cluster.Score - initialScore)
		}

		avgScore := 0.0
		if totalPrograms > 0 {
			avgScore = totalScore / float64(totalPrograms)
		}

		bestProgram := island.BestProgram
		bestSkeleton := "N/A"
		if bestProgram != nil {
			bestSkeleton = bestProgram.Skeleton
		}

		t.Logf("  Island %d: %d clusters, %d programs, Evals: %d, Culls: %d, Avg Score: %.2f, Best Score: %.2f, Best Skeleton: '%s'",
			island.ID, len(island.Clusters), totalPrograms, island.EvaluationsCount, island.CullingCount, avgScore, island.BestProgram.Score, bestSkeleton)
	}
	t.Logf("Total Propose-Weighted Sum: %.2f", totalProposeWeightedSum)

	var sequenceBuilder strings.Builder
	issueCount := 0
	updateCount := 0
	for _, event := range trace {
		if len(event.Type) > 0 {
			sequenceBuilder.WriteByte(event.Type[0])
		}
		switch event.Type {
		case bilevel.CallIssue:
			issueCount++
		case bilevel.CallUpdate:
			updateCount++
		}
	}
	sequence := sequenceBuilder.String()
	if len(sequence) > 100 {
		sequence = sequence[:100] + "..."
	}
	t.Logf("Call Sequence: %s", sequence)
	t.Logf("Total Calls: Issue=%d, Update=%d", issueCount, updateCount)
	t.Logf("Initial score: %f, Best score found: %f", initialScore, getBestScore(state))
	t.Log("---------------------")
}

func getBestScore(s *DeterministicState) ProgramScore {
	bestScore := ProgramScore(-1e9)
	for _, island := range s.Islands {
		islandBest := island.BestProgram.Score
		if islandBest > bestScore {
			bestScore = islandBest
		}
	}
	return bestScore
}
