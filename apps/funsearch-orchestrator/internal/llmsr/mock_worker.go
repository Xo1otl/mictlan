package llmsr

import (
	"context"
	"fmt"
	"strconv"
)

func NewScoreFromString(s string) (ProgramScore, error) {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, err
	}
	return f, nil
}

func MockPropose(ctx context.Context, req ProposeRequest) ProposeResult {
	if len(req.Parents) == 0 {
		return ProposeResult{Err: fmt.Errorf("%w: no parents provided", ErrInPropose)}
	}

	bestParent := req.Parents[0]
	for _, p := range req.Parents[1:] {
		if p.Score > bestParent.Score {
			bestParent = p
		}
	}

	parentSkeleton := bestParent.Skeleton
	val, err := NewScoreFromString(parentSkeleton)
	if err != nil {
		return ProposeResult{Err: fmt.Errorf("%w: invalid parent skeleton: %w", ErrInPropose, err)}
	}

	newSkeletons := []Skeleton{
		strconv.Itoa(int(val) + 1),
		strconv.Itoa(int(val) + 1),
	}

	return ProposeResult{
		Skeletons: newSkeletons,
		Metadata:  Metadata{IslandID: req.IslandID},
	}
}

func MockObserve(ctx context.Context, req ObserveRequest) ObserveResult {
	if req.Err != nil {
		return ObserveResult{
			Metadata: req.Metadata,
			Err:      req.Err,
		}
	}

	score, err := NewScoreFromString(req.Query)
	if err != nil {
		return ObserveResult{
			Query:    req.Query,
			Metadata: req.Metadata,
			Err:      fmt.Errorf("%w: invalid skeleton: %w", ErrInObserve, err),
		}
	}

	return ObserveResult{
		Query:    req.Query,
		Evidence: score,
		Metadata: req.Metadata,
	}
}
