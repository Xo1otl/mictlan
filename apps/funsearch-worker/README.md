# Introduction

The `orchestrator` is implemented in Go to manage the search process, while the computationally intensive tasks of program generation and numerical optimization are handled by Python libraries like JAX and NumPy. `funsearch-worker` serves as a dedicated computational service to bridge this language gap, providing a safe and efficient interface between the two environments.

# Overview

`funsearch-worker` is a Python-based gRPC server that exposes two independent RPC endpoints: the **`Propose` Service** and the **`Observe` Service**. This document also details the **Project Structure** and **Usage**. The separation of services allows the Go-based `orchestrator` to call only the necessary computational service for each phase of the search, keeping the core logic clean and focused.

# Project Structure

The `funsearch-worker` is structured like below.

```
.
├── api/
├── src/
│   └── funsearch_worker/
│       └── __init__.py
├── pyproject.toml
└── README.md
```

- **`api/`**: Contains Protocol Buffers (`.proto`) files that define the gRPC service contracts for the `Propose` and `Observe` services, including their request and response message types.
- **`src/funsearch_worker/`**: Contains the core application logic and implements the gRPC services defined in the `api/` directory.

# `Propose` Service

The `Propose` service is called by the `orchestrator` to generate a new program candidate. It receives contextual information, such as existing programs and the problem definition, and constructs a prompt for a Large Language Model (LLM). After executing inference, it parses the LLM's response and returns an executable program skeleton to the `orchestrator`.

# `Observe` Service

The `Observe` service is called by the `orchestrator` to evaluate a program candidate's performance. It receives a program skeleton and an evaluation dataset. Using numerical computation libraries like JAX and SciPy, it optimizes the parameters within the skeleton against the dataset. Finally, it calculates a performance score for the optimized program and returns it to the `orchestrator`.

# Usage
To start the gRPC server, run the following command from the project's root directory:

```sh
uv run funsearch-worker
```
