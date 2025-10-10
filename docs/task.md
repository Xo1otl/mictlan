# Objective
Define a generalized protocol for the `propose` step in evolutionary algorithms.

# Core Task
Decompose the `propose` function into a sequence of abstract, universal stages. The goal is to create a conceptual framework, not a specific implementation. This protocol should serve as a universal blueprint applicable to a wide range of evolutionary search methods.

# Key Requirements
The resulting protocol must be general enough to accommodate, but not be limited by, the following variations:

* **Parent Selection:** Handling single or multiple parents as input.
* **Modification Strategy:** Applying modifications as either diffs/patches or complete rewrites.
* **Candidate Generation:** Producing single or multiple candidate solutions as output.

Your definition should identify the fundamental concepts and their interactions within the `propose` step (e.g., parent selection, modification generation, candidate construction) to establish a truly generalized procedure.
