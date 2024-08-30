import Lake
open Lake DSL

package «leantutorial»

require mathlib from git "https://github.com/leanprover-community/mathlib4"

lean_exe «dependenttypetheory» where
  root := `examples.«TheoremProvingInLean4».DependentTypeTheory

lean_exe «propositionsandproofs» where
  root := `examples.«TheoremProvingInLean4».PropositionsAndProofs
