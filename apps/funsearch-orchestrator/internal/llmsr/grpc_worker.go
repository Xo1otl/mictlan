package llmsr

import (
	"context"
	"fmt"

	"funsearch-orchestrator/internal/pb"
)

// TODO: skeletonを送ってるだけだが、シグネチャも必要なのか？それともdraftという名前で変更適用対象としてのoriginalを含めるか。
func NewGRPCPropose(client pb.FUNSEARCHClient) func(context.Context, ProposeRequest) ProposeResult {
	return func(ctx context.Context, req ProposeRequest) ProposeResult {
		pbParents := make([]*pb.Candidate, len(req.Parents))
		for i, p := range req.Parents {
			pbParents[i] = &pb.Candidate{Hypothesis: string(p.Skeleton), Quantitative: p.Score}
		}

		pbReq := &pb.ProposeRequest{Parents: pbParents}
		resp, err := client.Propose(ctx, pbReq)
		if err != nil {
			return ProposeResult{Err: fmt.Errorf("%w: gRPC propose error: %w", ErrInPropose, err)}
		}

		skeletons := make([]Skeleton, len(resp.Hypothesises))
		for i, s := range resp.Hypothesises {
			skeletons[i] = Skeleton(s)
		}

		return ProposeResult{
			Skeletons: skeletons,
			Metadata:  Metadata{IslandID: req.IslandID},
		}
	}
}

func NewGRPCObserve(client pb.FUNSEARCHClient) func(context.Context, ObserveRequest) ObserveResult {
	return func(ctx context.Context, req ObserveRequest) ObserveResult {
		if req.Err != nil {
			return ObserveResult{
				Metadata: req.Metadata,
				Err:      req.Err,
			}
		}

		pbReq := &pb.ObserveRequest{Hypothesis: string(req.Query)}
		resp, err := client.Observe(ctx, pbReq)
		if err != nil {
			return ObserveResult{
				Query:    req.Query,
				Metadata: req.Metadata,
				Err:      fmt.Errorf("%w: gRPC observe error: %w", ErrInObserve, err),
			}
		}

		return ObserveResult{
			Query:    Skeleton(resp.Hypothesis),
			Evidence: resp.Quantitative,
			Metadata: req.Metadata,
		}
	}
}
