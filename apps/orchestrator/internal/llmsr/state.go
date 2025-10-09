package llmsr

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"time"
)

// DeterministicState holds the complete state of the island model evolutionary algorithm.
//
// The design assumes a single-threaded execution model where all method calls are serialized by a controlling goroutine.
// All code paths must be strictly deterministic.
type DeterministicState struct {
	Islands               map[int]*Island
	MaxEvaluations        int
	EvaluationsCount      int
	MigrationInterval     int
	NextMigration         int
	InitialSkeleton       Skeleton
	NumIslandsToEliminate int
	ScoreQuantization     int
	rng                   *rand.Rand
	T0                    float64
	N                     int
	Tp                    float64
}

func NewDeterministicState(initialSkeleton Skeleton, initialScore ProgramScore, maxEvaluations, numIslands, migrationInterval, scoreQuantization int, eliminationRate float64, t0 float64, n int, tp float64, rng *rand.Rand) (*DeterministicState, error) {
	if eliminationRate < 0 || eliminationRate >= 1 {
		return nil, fmt.Errorf("%w: elimination rate must be in [0, 1), got %f", ErrInvalidParameter, eliminationRate)
	}
	if n <= 0 {
		return nil, fmt.Errorf("%w: N must be positive, got %d", ErrInvalidParameter, n)
	}
	if t0 < 0 {
		return nil, fmt.Errorf("%w: T0 must be non-negative, got %f", ErrInvalidParameter, t0)
	}
	if tp <= 0 {
		return nil, fmt.Errorf("%w: Tp must be positive, got %f", ErrInvalidParameter, tp)
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	s := &DeterministicState{
		Islands:               make(map[int]*Island, numIslands),
		MaxEvaluations:        maxEvaluations,
		MigrationInterval:     migrationInterval,
		NextMigration:         migrationInterval,
		InitialSkeleton:       initialSkeleton,
		NumIslandsToEliminate: int(float64(numIslands) * eliminationRate),
		ScoreQuantization:     scoreQuantization,
		rng:                   rng,
		T0:                    t0,
		N:                     n,
		Tp:                    tp,
	}

	initialClusterScore, err := quantize(initialScore, s.ScoreQuantization)
	if err != nil {
		return nil, err
	}

	for i := range numIslands {
		program := &Program{Skeleton: initialSkeleton, Score: initialScore}
		cluster := &Cluster{Score: initialClusterScore, Programs: []*Program{program}}
		s.Islands[i] = &Island{
			ID:             i,
			Clusters:       map[ClusterScore]*Cluster{initialClusterScore: cluster},
			PopulationSize: 1,
			BestProgram:    program,
		}
	}
	return s, nil
}

func (s *DeterministicState) Update(res ObserveResult) (done bool, err error) {
	island, ok := s.Islands[res.Metadata.IslandID]
	if !ok {
		return true, fmt.Errorf("%w: island with ID %d not found", ErrIslandNotFound, res.Metadata.IslandID)
	}

	if island.PendingObservations != 0 {
		if island.PendingObservations == -1 {
			if res.Metadata.NumSiblings > 1 {
				island.PendingObservations = res.Metadata.NumSiblings - 1
			} else {
				island.PendingObservations = 0
			}
		} else {
			island.PendingObservations--
		}
	}

	if res.Err != nil {
		// Errors still count as an evaluation against the cap, but don't update island state.
		s.EvaluationsCount++
		return s.EvaluationsCount >= s.MaxEvaluations, nil
	}
	s.EvaluationsCount++

	program := &Program{Skeleton: res.Query, Score: res.Evidence}
	if err := island.addProgram(program, s.ScoreQuantization); err != nil {
		return true, err
	}

	if s.EvaluationsCount >= s.NextMigration {
		if err := s.manageIslands(); err != nil {
			return true, err
		}
		s.NextMigration += s.MigrationInterval
	}
	return s.EvaluationsCount >= s.MaxEvaluations, nil
}

func (s *DeterministicState) Issue() (ProposeRequest, bool, error) {
	availableIslandIDs := make([]int, 0, len(s.Islands))
	for id, island := range s.Islands {
		if island.PendingObservations == 0 {
			availableIslandIDs = append(availableIslandIDs, id)
		}
	}

	if len(availableIslandIDs) == 0 {
		return ProposeRequest{}, false, nil
	}
	sort.Ints(availableIslandIDs)

	randomID := availableIslandIDs[s.rng.Intn(len(availableIslandIDs))]
	island := s.Islands[randomID]

	if len(island.Clusters) == 0 {
		for _, otherIsland := range s.Islands {
			if len(otherIsland.Clusters) > 0 {
				return ProposeRequest{}, false, fmt.Errorf("%w: island %d is empty, but other islands are not", ErrEmptyIslandSelected, island.ID)
			}
		}
	}

	parent1, err := s.selectParent(island)
	if err != nil {
		return ProposeRequest{}, false, err
	}
	parent2, err := s.selectParent(island)
	if err != nil {
		return ProposeRequest{}, false, err
	}

	island.PendingObservations = -1

	return ProposeRequest{
		Parents:  []*Program{parent1, parent2},
		IslandID: island.ID,
	}, true, nil
}

func (s *DeterministicState) selectParent(island *Island) (*Program, error) {
	selectedCluster, err := s.selectCluster(island)
	if err != nil {
		return nil, err
	}
	return s.selectProgramFromCluster(selectedCluster, island.ID)
}

func (s *DeterministicState) selectCluster(island *Island) (*Cluster, error) {
	if len(island.Clusters) == 0 {
		return nil, fmt.Errorf("%w: cannot select cluster from empty island %d", ErrSelectionFromEmptyIsland, island.ID)
	}

	clusterScores := make([]ClusterScore, 0, len(island.Clusters))
	for score := range island.Clusters {
		clusterScores = append(clusterScores, score)
	}
	sort.Float64s(clusterScores)

	clusters := make([]*Cluster, 0, len(island.Clusters))
	maxClusterScore := ClusterScore(math.Inf(-1))
	for _, score := range clusterScores {
		cluster := island.Clusters[score]
		clusters = append(clusters, cluster)
		if cluster.Score > maxClusterScore {
			maxClusterScore = cluster.Score
		}
	}

	tc := s.T0*(1-float64(island.PopulationSize%s.N)/float64(s.N)) + Epsilon

	clusterWeightFunc := func(c *Cluster) float64 {
		return math.Exp((c.Score - maxClusterScore) / tc)
	}
	selectedCluster, err := weightedChoice(clusters, clusterWeightFunc, s.rng)
	if err != nil {
		return nil, fmt.Errorf("%w in island %d: %w", ErrClusterSelectionFailed, island.ID, err)
	}
	return selectedCluster, nil
}

func (s *DeterministicState) selectProgramFromCluster(cluster *Cluster, islandID int) (*Program, error) {
	programs := cluster.Programs
	if len(programs) == 0 {
		return nil, fmt.Errorf("%w: cannot select program from empty cluster in island %d", ErrInvalidCluster, islandID)
	}
	if len(programs) == 1 {
		return programs[0], nil
	}

	minLength, maxLength := math.MaxInt32, 0
	for _, p := range programs {
		l := len(p.Skeleton)
		if l < minLength {
			minLength = l
		}
		if l > maxLength {
			maxLength = l
		}
	}

	lengthRange := float64(maxLength-minLength) + Epsilon
	skeletonWeightFunc := func(p *Program) float64 {
		normalizedLength := float64(len(p.Skeleton)-minLength) / lengthRange
		return math.Exp(-normalizedLength / s.Tp)
	}
	selectedProgram, err := weightedChoice(programs, skeletonWeightFunc, s.rng)
	if err != nil {
		return nil, fmt.Errorf("%w from cluster with score %f in island %d: %w", ErrProgramSelectionFailed, cluster.Score, islandID, err)
	}

	return selectedProgram, nil
}

func (s *DeterministicState) manageIslands() error {
	if len(s.Islands) <= s.NumIslandsToEliminate {
		return nil
	}

	allIslands := make([]*Island, 0, len(s.Islands))
	for _, island := range s.Islands {
		allIslands = append(allIslands, island)
	}

	sort.Slice(allIslands, func(i, j int) bool {
		scoreI := allIslands[i].BestProgram.Score
		scoreJ := allIslands[j].BestProgram.Score
		if scoreI != scoreJ {
			return scoreI > scoreJ
		}
		return allIslands[i].ID < allIslands[j].ID
	})

	numSurvivors := len(allIslands) - s.NumIslandsToEliminate
	survivors := allIslands[:numSurvivors]
	culled := allIslands[numSurvivors:]

	if len(survivors) == 0 || survivors[len(survivors)-1].BestProgram == nil {
		return ErrNoElitesFound
	}

	for _, islandToReplace := range culled {
		randomSurvivor := survivors[s.rng.Intn(len(survivors))]
		elite := randomSurvivor.BestProgram
		if err := islandToReplace.resetWithElite(elite, s.ScoreQuantization); err != nil {
			return err
		}
	}
	return nil
}

func weightedChoice[T any](items []T, getWeight func(T) float64, rng *rand.Rand) (T, error) {
	var zero T
	if len(items) == 0 {
		return zero, ErrSelectionFromEmptySlice
	}

	weights := make([]float64, len(items))
	sumWeights := 0.0
	for i, item := range items {
		w := getWeight(item)
		if w < 0 {
			return zero, ErrNegativeWeight
		}
		weights[i] = w
		sumWeights += w
	}

	if sumWeights <= Epsilon {
		return zero, ErrNumericalInstability
	}

	randVal := rng.Float64() * sumWeights
	cumulativeWeight := 0.0
	for i, w := range weights {
		cumulativeWeight += w
		if randVal <= cumulativeWeight {
			return items[i], nil
		}
	}
	return items[len(items)-1], nil
}

type Island struct {
	ID                  int
	Clusters            map[ClusterScore]*Cluster
	PopulationSize      int
	EvaluationsCount    int
	CullingCount        int
	BestProgram         *Program
	PendingObservations int
}

func (i *Island) addProgram(p *Program, quantization int) error {
	i.EvaluationsCount++
	if p.isBetterThan(i.BestProgram) {
		i.BestProgram = p
		clusterScore, err := quantize(p.Score, quantization)
		if err != nil {
			return err
		}

		if cluster, ok := i.Clusters[clusterScore]; ok {
			cluster.Programs = append(cluster.Programs, p)
		} else {
			i.Clusters[clusterScore] = &Cluster{Score: clusterScore, Programs: []*Program{p}}
		}
		i.PopulationSize++
	}
	return nil
}

func (i *Island) resetWithElite(elite *Program, quantization int) error {
	clusterScore, err := quantize(elite.Score, quantization)
	if err != nil {
		return err
	}
	i.Clusters = map[ClusterScore]*Cluster{clusterScore: {Score: clusterScore, Programs: []*Program{elite}}}
	i.PopulationSize = 1
	i.EvaluationsCount = 0
	i.CullingCount++
	i.BestProgram = elite
	return nil
}

type Cluster struct {
	Score    ClusterScore
	Programs []*Program
}

type ProposeRequest struct {
	Parents  []*Program
	IslandID int
}

type ObserveResult struct {
	Query    Skeleton
	Evidence ProgramScore
	Metadata Metadata
	Err      error
}

type Metadata struct {
	IslandID    int
	NumSiblings int
}

func quantize(score ProgramScore, precision int) (ClusterScore, error) {
	key := strconv.FormatFloat(score, 'f', precision, 64)
	f, err := strconv.ParseFloat(key, 64)
	if err != nil {
		return 0, fmt.Errorf("%w: failed to parse '%s': %w", ErrQuantization, key, err)
	}
	return f, nil
}

type Program struct {
	Skeleton Skeleton
	Score    ProgramScore
}

func (p *Program) isBetterThan(other *Program) bool {
	if p.Score != other.Score {
		return p.Score > other.Score
	}
	if len(p.Skeleton) != len(other.Skeleton) {
		return len(p.Skeleton) < len(other.Skeleton)
	}
	return false
}

type Skeleton = string

type ProgramScore = float64
type ClusterScore = float64

const (
	Epsilon = 1e-9
)
