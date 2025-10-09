package rastrigin_test

import (
	"context"
	"testing"
	"time"

	"funsearch-orchestrator/internal/bilevel"
	"funsearch-orchestrator/internal/rastrigin"

	"github.com/stretchr/testify/assert"
)

func TestRastriginWithRunner(t *testing.T) {
	const (
		islandPopulation   = 50
		numIslands         = 5
		totalEvaluations   = 250000
		migrationInterval  = 25
		migrationSize      = 5
		proposeConcurrency = 2
		observeConcurrency = 4
		testTimeout        = 5 * time.Second
	)

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	state := rastrigin.NewState(
		islandPopulation,
		numIslands,
		totalEvaluations,
		migrationInterval,
		migrationSize,
	)

	orchestrator := bilevel.NewOrchestrator(
		rastrigin.Propose,
		rastrigin.Observe,
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

	bilevel.Run(orchestrator, ctx, state, errCh)

	if ctx.Err() == context.DeadlineExceeded {
		t.Fatal("Test timed out, indicating a potential deadlock.")
	}

	var bestFitness rastrigin.Fitness = 1e6
	for _, island := range state.Islands {
		for _, individual := range island.Population {
			if individual.Fitness < bestFitness {
				bestFitness = individual.Fitness
			}
		}
	}

	assert.True(t, state.EvaluationsCount >= totalEvaluations, "Should have completed at least the specified number of evaluations")
	assert.Less(t, bestFitness, 1.0, "The final best fitness should be less than 1.0")

	t.Logf("Test finished. Best fitness found: %f", bestFitness)
}
