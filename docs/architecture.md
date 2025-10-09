# Introduction

In recent years, approaches that use Large Language Models (LLMs) to automate scientific discovery, such as Alpha Evolve and Deep Researcher with Test-Time Diffusion, have achieved success. This project, "Alpha Probe," builds on this trend, aiming to construct a framework specialized in the discovery of interpretable theoretical models (e.g., mathematical formulas and differential equations), particularly in the fields of chemistry and physics. The goal is to find universal laws and equations that humans can understand, rather than black-box prediction models from machine learning.

# Overview

The system architecture of Alpha Probe is designed using the C4 model. This document will explain the system in stages, from a high-level overview to the finer details. First, we will define the **System Context**, which shows how the system interacts with its external environment. Next, we will present the **Container Architecture**, which depicts the main services that make up the system. Finally, we will detail the internal structure of the **Command Service Component Architecture**, which executes the core discovery process.

# System Context

The System Context diagram illustrates the relationship between the Alpha Probe framework, the "Researcher" who uses it, and the external systems it interacts with. The researcher executes model discovery through Alpha Probe, and the system internally utilizes external computational resources such as LLMs and GPUs.

```plantuml
@startuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Context.puml

' Diagram Title
title Level 1: System Context Diagram for Alpha Probe

' Element Definitions
Person(researcher, "Researcher", "User who discovers chemical/physical models")
System(alpha_probe, "Alpha Probe Framework", "Library for discovering mathematical/theoretical models")
System_Ext(llm, "LLMs and GPUs", "External foundational models used for hypothesis generation and evaluation")
System_Ext(external_analysis, "External Analysis Platform", "Platform for analysis and data storage (Optional)")

' Relationship Definitions
Rel(researcher, alpha_probe, "Executes discovery, analyzes results")
Rel(alpha_probe, llm, "Requests hypothesis generation/evaluation via API")
Rel(alpha_probe, external_analysis, "Transfers analysis data")

@enduml
```

# Container Architecture

The Container diagram visualizes the main services (containers) that constitute the Alpha Probe framework. A UI that accepts user requests, a Command Service that governs the discovery process, various services for data transformation and querying, and a real-time log streaming infrastructure all work in concert.

```plantuml
@startuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml

' Diagram Title
title Level 2: Container Diagram for Alpha Probe

' External Element Definitions
Person(researcher, "Researcher", "Conducts research using the Alpha Probe framework")
System_Ext(llm, "LLMs and GPUs", "External computational resources and large language models")

' Client-side System Boundary
System_Boundary(client, "Researcher's PC/Browser") {
    Container(spa, "Web UI", "Single-Page App", "JavaScript, React/Vue, etc. Provides the user interface.")
}

' Server-side System Boundary
System_Boundary(alpha_probe, "Alpha Probe Framework (Backend)") {
    Container(command_service, "Command Service", "Backend", "Executes the discovery process. Generates logs.")
    Container(projection_service, "Projection Service", "Data Processor", "Transforms and transfers state data.")
    Container(query_service, "Query Service", "API Service", "Executes queries against analysis data.")

    ' --- Add containers for real-time logging ---
    Container(realtime_service, "Real-time Service", "Real-time Push Protocol", "Manages client connections and streams logs from the Broker.")
    ContainerQueue(message_broker, "Message Broker", "Durable Pub/Sub (e.g., Redpanda is lightweight.)", "Log message persistence and message bus.")
    ' -----------------------------------------

    ContainerDb(primary_db, "Primary Datastore", "MongoDB", "Persists the state of the discovery process.")
    ContainerDb(analysis_db, "Analysis Datastore", "DWH/VDB/RDP", "Data optimized for analysis and visualization.")
}

' Relationship Definitions
' Researcher uses the SPA in the browser
Rel_D(researcher, spa, "Uses", "HTTPS")

' SPA communicates with backend APIs
Rel(spa, command_service, "Discovery Request", "JSON/HTTPS")
Rel(spa, query_service, "Result Inquiry", "JSON/HTTPS")

' --- Add relationships for real-time logging ---
' 1. UI establishes a persistent connection with the Real-time Service
Rel(spa, realtime_service, "Subscribe to logs", "WSS / HTTPS")

' 2. Command Service publishes logs to the Message Broker
Rel_R(command_service, message_broker, "Publishes logs", "Async API")

' 3. Real-time Service subscribes to logs from the Message Broker
Rel_L(realtime_service, message_broker, "Subscribes to logs")
' -----------------------------------------


' Internal backend communication
Rel_D(command_service, primary_db, "R/W", "State persistence and restoration")
Rel_L(command_service, llm, "API Request", "Hypothesis generation/evaluation")

Rel(primary_db, projection_service, "Push (Change Streams)", "Notifies of changes")
Rel_D(projection_service, analysis_db, "Write", "Saves transformed data")

Rel(query_service, analysis_db, "Query", "Data inquiry")
@enduml
```

# Command Service Component Architecture

The Command Service is the core component responsible for executing the model discovery process. Its internal structure is designed as a ring architecture based on Go's CSP (Communicating Sequential Processes) model. In this architecture, components with the responsibilities of "State," "Propose," "Adapter," and "Observe" are connected in a ring, processing data as it circulates through channels. This enables safe and efficient parallel processing and state management.

```plantuml
@startuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Component.puml

title Level 3: Component Diagram for Command Service (Ring Architecture)

Container(ui_service, "UI Service", "Web UI")
ContainerDb(primary_db, "Primary DB", "Persistence")
System_Ext(external_services, "External Services", "LLM, GPU, etc.")

Container_Boundary(command_service, "Command Service") {
    Component(application, "Application", "Entry point. Creates the Orchestrator, builds and runs the pipeline ring.")
    Component(repository, "Repository", "Persistence Layer (Caretaker)")

    Boundary(pipeline_ring, "Pipeline Ring (bilevel execution model)") {
        Component(state, "State Controller", "GoController", "Manages state. Generates Propose tasks and processes Observe results.")
        Component(propose, "Propose Workers", "GoWorkers", "Executes Propose tasks in parallel.")
        Component(adapter, "Adapter Controller", "GoController", "Aggregates and transforms Propose results, generates Observe tasks.")
        Component(observe, "Observe Workers", "GoWorkers", "Executes Observe tasks in parallel.")
    }
}

' --- Initialization Flow ---
Rel(ui_service, application, "1. Search Request")
Rel(application, repository, "2. Instruct State Load")
Rel(repository, primary_db, "Read")
Rel(repository, state, "3. Restore State")
Rel(application, pipeline_ring, "4. Start Pipeline")


' --- Pipeline Data Flow (Ring Architecture) ---
Rel(state, propose, "Sends Propose request")
Rel(propose, adapter, "Sends Propose result")
Rel(adapter, observe, "Sends Observe request")
Rel(observe, state, "Sends Observe result")


' --- Persistence ---
Rel(application, repository, "Save State (Periodically)")
Rel(repository, state, "Create Memento")
Rel_Back(state, repository, "Persist State")


' --- External Service Usage ---
Rel(propose, external_services, "Uses")
Rel(observe, external_services, "Uses")

@enduml
```