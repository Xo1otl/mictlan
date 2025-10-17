# API Design Ideas

## Design Goal

The goal is to design a flexible API for running simulations. The API must handle general patterns of execution based on which parameters are fixed versus which are being swept.

The primary execution patterns are:
1.  **Single Run**: All parameters (`superlattice`, `delta_k_pair`, `b_initial`) are single, fixed values for a single simulation.
2.  **1D Sweep**: One parameter is an array of values to be iterated over, while all other parameters remain fixed.
3.  **N-D Sweep (Future Possibility)**: Two or more parameters are arrays of values, defining a grid of simulations to be executed.

The API must handle the **Single Run** and **1D Sweep** patterns, while being extensible enough to handle **N-D Sweeps** in the future without requiring a fundamental redesign.

# **Task**
Consider better approach.
