package rastrigin

import (
	"context"
	"math"
	"math/rand"
	"sort"
)

type Gene = []float64
type Fitness = float64
type Population = []Individual

type Individual struct {
	Gene    Gene
	Fitness Fitness
}

type Island struct {
	ID         int
	Population Population
}

type Metadata struct {
	IslandID int
}

type State struct {
	Islands            []*Island
	PendingIslands     map[int]bool
	AvailableIslandIDs []int
	EvaluationsCount   int
	TotalEvaluations   int
	MigrationInterval  int
	MigrationSize      int
}

func NewState(
	islandPopulation int,
	numIslands int,
	totalEvaluations int,
	migrationInterval int,
	migrationSize int,
) *State {
	islands := make([]*Island, numIslands)
	for i := range numIslands {
		islands[i] = &Island{ID: i, Population: newInitialPopulation(islandPopulation)}
	}

	availableIDs := make([]int, len(islands))
	for i, island := range islands {
		availableIDs[i] = island.ID
	}

	return &State{
		Islands:            islands,
		PendingIslands:     make(map[int]bool),
		AvailableIslandIDs: availableIDs,
		EvaluationsCount:   0,
		TotalEvaluations:   totalEvaluations,
		MigrationInterval:  migrationInterval,
		MigrationSize:      migrationSize,
	}
}

func (s *State) Update(res ObserveResult) (done bool, err error) {
	islandID := res.Metadata.IslandID
	evaluatedChild := Individual{Gene: res.Gene, Fitness: res.Fitness}

	delete(s.PendingIslands, islandID)
	s.EvaluationsCount++
	s.AvailableIslandIDs = append(s.AvailableIslandIDs, islandID)

	incorporate(s.Islands[islandID], []Individual{evaluatedChild})

	if s.EvaluationsCount > 0 && s.EvaluationsCount%s.MigrationInterval == 0 {
		migrate(s.Islands, s.MigrationSize)
	}

	return s.EvaluationsCount >= s.TotalEvaluations, nil
}

func (s *State) Issue() (*Island, bool, error) {
	if len(s.AvailableIslandIDs) == 0 {
		return nil, false, nil
	}
	randIndex := rand.Intn(len(s.AvailableIslandIDs))
	islandID := s.AvailableIslandIDs[randIndex]

	// Remove the ID from AvailableIslandIDs
	s.AvailableIslandIDs = append(s.AvailableIslandIDs[:randIndex], s.AvailableIslandIDs[randIndex+1:]...)
	// Add the ID to PendingIslands
	s.PendingIslands[islandID] = true

	return s.Islands[islandID], true, nil
}

// --- Types for bilevel Runner ---

type ObserveRequest struct {
	Gene     Gene
	Metadata Metadata
}

type ObserveResult struct {
	Gene     Gene
	Fitness  Fitness
	Metadata Metadata
}

// --- Pipeline Functions ---

func Propose(ctx context.Context, island *Island) ObserveRequest {
	pop := island.Population
	tournament := func() Individual {
		best := pop[rand.Intn(len(pop))]
		for i := 1; i < 5; i++ { // TournamentSize
			competitor := pop[rand.Intn(len(pop))]
			if competitor.Fitness < best.Fitness {
				best = competitor
			}
		}
		return best
	}
	parent1, parent2 := tournament(), tournament()

	var childGene Gene
	if rand.Float64() < 0.9 {
		childGene = crossoverBLXAlpha(parent1.Gene, parent2.Gene, 0.5)
	} else {
		childGene = make(Gene, 30)
		copy(childGene, parent1.Gene)
	}

	for i := range childGene {
		if rand.Float64() < 1.0/30.0 {
			childGene[i] += rand.NormFloat64() * ((5.12 - (-5.12)) * 0.05)
			childGene[i] = math.Max(-5.12, math.Min(5.12, childGene[i]))
		}
	}

	return ObserveRequest{Gene: childGene, Metadata: Metadata{IslandID: island.ID}}
}

func Observe(ctx context.Context, req ObserveRequest) ObserveResult {
	a := 10.0
	sum := a * float64(len(req.Gene))
	for _, x := range req.Gene {
		sum += x*x - a*math.Cos(2*math.Pi*x)
	}
	return ObserveResult{Gene: req.Gene, Fitness: Fitness(sum), Metadata: req.Metadata}
}

func newInitialPopulation(size int) Population {
	pop := make(Population, size)
	for i := range pop {
		gene := make(Gene, 30)
		for j := range gene {
			gene[j] = -5.12 + rand.Float64()*(5.12-(-5.12))
		}
		fitness := Observe(context.Background(), ObserveRequest{Gene: gene})
		pop[i] = Individual{Gene: gene, Fitness: fitness.Fitness}
	}
	return pop
}

func crossoverBLXAlpha(p1, p2 Gene, alpha float64) Gene {
	child := make(Gene, len(p1))
	for i := range p1 {
		d := math.Abs(p1[i] - p2[i])
		minGene := math.Min(p1[i], p2[i]) - alpha*d
		maxGene := math.Max(p1[i], p2[i]) + alpha*d
		child[i] = minGene + rand.Float64()*(maxGene-minGene)
	}
	return child
}

func incorporate(island *Island, individuals []Individual) {
	sort.Slice(island.Population, func(a, b int) bool {
		return island.Population[a].Fitness > island.Population[b].Fitness
	})
	for j := 0; j < len(individuals) && j < len(island.Population); j++ {
		if individuals[j].Fitness < island.Population[j].Fitness {
			island.Population[j] = individuals[j]
		}
	}
}

func migrate(islands []*Island, migrationSize int) {
	if len(islands) <= 1 {
		return
	}
	allMigrants := make([][]Individual, len(islands))
	for i, island := range islands {
		sort.Slice(island.Population, func(a, b int) bool {
			return island.Population[a].Fitness < island.Population[b].Fitness
		})
		count := min(migrationSize, len(island.Population))
		allMigrants[i] = island.Population[:count]
	}
	for i, sourceIslandMigrants := range allMigrants {
		targetIslandIndex := (i + 1) % len(islands)
		incorporate(islands[targetIslandIndex], sourceIslandMigrants)
	}
}
