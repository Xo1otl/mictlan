package llmsr

import "errors"

var (
	ErrInObserve = errors.New("error occurred in observe phase")
	ErrInPropose = errors.New("error occurred in propose phase")

	// --- State Management Errors ---

	ErrIslandNotFound           = errors.New("island not found")
	ErrEmptyIslandSelected      = errors.New("selected an empty island while others are populated")
	ErrSelectionFromEmptyIsland = errors.New("parent selection from an empty island")
	ErrNoElitesFound            = errors.New("no elites found from surviving islands")
	ErrInvalidParameter         = errors.New("invalid parameter provided")
	ErrNoSkeletonsGenerated     = errors.New("propose returned no skeletons")

	// --- Selection Process Errors ---

	ErrClusterSelectionFailed  = errors.New("cluster selection failed")
	ErrProgramSelectionFailed  = errors.New("program selection failed")
	ErrInvalidCluster          = errors.New("selected cluster is invalid (nil or empty)")
	ErrSelectionFromEmptySlice = errors.New("selection from empty slice")
	ErrNegativeWeight          = errors.New("negative weight provided for selection")

	// --- Numerical Stability Errors ---

	ErrNumericalInstability = errors.New("numerical instability detected")
	ErrQuantization         = errors.New("failed to quantize score")
)
