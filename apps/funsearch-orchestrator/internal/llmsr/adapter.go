package llmsr

type Adapter struct {
	queue []ObserveRequest
}

func NewAdapter() *Adapter {
	return &Adapter{
		queue: make([]ObserveRequest, 0),
	}
}

func (a *Adapter) Recv(res ProposeResult) {
	err := res.Err
	if err == nil && len(res.Skeletons) == 0 {
		err = ErrNoSkeletonsGenerated
	}

	skeletons := res.Skeletons
	if len(skeletons) == 0 {
		skeletons = []Skeleton{""} // Add a dummy skeleton to carry the error.
	}

	numSiblings := len(skeletons)

	for i, skeleton := range skeletons {
		req := ObserveRequest{
			Query:    skeleton,
			Metadata: Metadata{IslandID: res.Metadata.IslandID, NumSiblings: numSiblings},
		}
		if i == 0 {
			req.Err = err
		}
		a.queue = append(a.queue, req)
	}
}

func (a *Adapter) Next() (ObserveRequest, bool) {
	if len(a.queue) == 0 {
		return ObserveRequest{}, false
	}
	req := a.queue[0]
	a.queue = a.queue[1:]
	return req, true
}

type ProposeResult struct {
	Skeletons []Skeleton
	Metadata  Metadata
	Err       error
}

type ObserveRequest struct {
	Query    Skeleton
	Metadata Metadata
	Err      error
}
