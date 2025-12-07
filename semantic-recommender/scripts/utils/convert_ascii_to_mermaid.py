#!/usr/bin/env python3
"""
Convert ASCII diagrams to Mermaid format in markdown files.
"""

import os
import re
from pathlib import Path

def convert_system_context_diagram():
    """Replace the broken system context diagram with proper Mermaid syntax."""
    return """```mermaid
graph TD
    subgraph External["EXTERNAL SYSTEMS"]
        CP["Content Providers"]
        UA["User Applications"]
        AI["AI Agents<br/>(Claude/etc)"]
    end

    subgraph Gateway["API GATEWAY LAYER"]
        REST["REST API Server<br/>(Actix-web/Axum)<br/>• Authentication<br/>• Rate Limiting<br/>• Request Routing<br/>• Response Cache"]
        MCP["MCP Server<br/>(JSON-RPC 2.0)<br/>• Tool Discovery<br/>• Streaming<br/>• Context Mgmt<br/>• Error Handling"]
    end

    subgraph Orchestration["ORCHESTRATION & ROUTING LAYER"]
        Router["Query Analyzer & Router<br/>• Complexity Analysis<br/>• Resource Estimation<br/>• Path Selection<br/>• Load Balancing<br/>• Fallback Strategy"]
    end

    subgraph GPU["GPU ENGINE LAYER<br/>(CUDA Kernels)"]
        Similarity["Semantic Similarity<br/>• Tensor Cores FP16<br/>• Memory Coalescing<br/>• Shared Memory Cache<br/>• 280 GB/s Bandwidth"]
        SSSP["Adaptive SSSP Engine<br/>• GPU Dijkstra<10K<br/>• Duan SSSP>10M<br/>• Auto crossover"]
    end

    subgraph VectorDB["VECTOR DATABASE LAYER<br/>(Qdrant/Milvus)"]
        HNSW["HNSW Index<br/>• M=32, efConstruction<br/>• Product Quantization<br/>• Disk-backed Storage<br/>• Horizontal Sharding"]
        MetaFilter["Metadata Filtering<br/>• Inverted Indices<br/>• Range Queries<br/>• Faceted Search"]
    end

    subgraph Enrichment["SEMANTIC ENRICHMENT LAYER"]
        KG["Knowledge Graph Neo4j<br/>• GMC-O Ontology Extended<br/>• Entity Relationships<br/>• APOC Procedures<br/>• Cypher Queries<br/>• Inference Rules"]
    end

    subgraph Learning["PERSONALIZATION & LEARNING LAYER"]
        AgentDB["Reinforcement Learning AgentDB<br/>• Thompson Sampling<br/>• User Context Embeddings<br/>• Exploration/Exploitation<br/>• Experience Replay<br/>• Cold-Start Handling"]
    end

    CP -->|Metadata Ingestion| REST
    UA -->|Queries REST/GraphQL| REST
    AI -->|MCP Protocol JSON-RPC| MCP

    REST --> Router
    MCP --> Router

    Router -->|GPU Path| GPU
    Router -->|Vector DB Path| VectorDB
    Router -->|Hybrid Path| GPU
    Router -->|Hybrid Path| VectorDB

    GPU -->|Scores| Enrichment
    VectorDB -->|Candidates| Enrichment

    Enrichment --> Learning

    Similarity --> SSSP

    HNSW --> MetaFilter

    style External fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style Gateway fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Orchestration fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style GPU fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    style VectorDB fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style Enrichment fill:#f0f4c3,stroke:#827717,stroke-width:2px
    style Learning fill:#ffccbc,stroke:#d84315,stroke-width:2px
```"""

def convert_deployment_topology():
    """Convert ASCII deployment topology to Mermaid."""
    return """```mermaid
graph TD
    LB["Load Balancer<br/>(NGINX/HAProxy)"]

    subgraph API["API Servers 3x"]
        AS1["API Server<br/>(Actix + MCP)"]
        AS2["API Server<br/>(Actix + MCP)"]
        AS3["API Server<br/>(Actix + MCP)"]
    end

    subgraph GPU["GPU Cluster"]
        GPU1["GPU Node 1<br/>(T4 GPU)"]
        GPU2["GPU Node 2<br/>(T4 GPU)"]
        GPUN["GPU Node N<br/>(T4 GPU)"]
    end

    subgraph Storage["Storage Layer"]
        QD["Qdrant Cluster<br/>(Sharded)"]
        NEO["Neo4j Cluster<br/>(Replicas)"]
        RD["Redis Cache"]
    end

    LB --> AS1
    LB --> AS2
    LB --> AS3

    AS1 --> GPU1
    AS2 --> GPU2
    AS3 --> GPUN

    GPU1 --> QD
    GPU2 --> NEO
    GPUN --> RD

    QD --> RD
    NEO --> RD

    style LB fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style API fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style GPU fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    style Storage fill:#f0f4c3,stroke:#827717,stroke-width:2px
```"""

def convert_multi_region_deployment():
    """Convert ASCII multi-region deployment to Mermaid."""
    return """```mermaid
graph LR
    subgraph US["Region 1 US-East"]
        USAPI["3x API Servers<br/>4x GPU Nodes<br/>Qdrant Shard 1<br/>Neo4j Replica 1"]
    end

    subgraph EU["Region 2 EU-West"]
        EUAPI["3x API Servers<br/>4x GPU Nodes<br/>Qdrant Shard 2<br/>Neo4j Replica 2"]
    end

    subgraph APAC["Region 3 APAC"]
        APACAPI["3x API Servers<br/>4x GPU Nodes<br/>Qdrant Shard 3<br/>Neo4j Replica 3"]
    end

    GR["Global Routing<br/>(GeoDNS)"]

    GR --> US
    GR --> EU
    GR --> APAC

    US <-->|Replication| EU
    EU <-->|Replication| APAC

    style US fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style EU fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style APAC fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style GR fill:#ffccbc,stroke:#d84315,stroke-width:2px
```"""

def convert_query_routing():
    """Convert ASCII query routing diagram to Mermaid."""
    return """```mermaid
flowchart TD
    Query["Query Received"]

    Parse["Parse & Validate<br/>Extract: text, filters, limit"]

    Estimate["Estimate Candidate Size<br/>Filter Selectivity Analysis"]

    Analysis["Complexity Analysis<br/>Resource Requirements<br/>GPU Memory Check"]

    Decision{"Routing Decision"}

    GPU_Path["GPU Path<br/>candidates < 10K<br/>Load → Kernel → Results<br/>< 10ms"]

    VDB_Path["Vector DB Path<br/>candidates > 100K<br/>HNSW → Filter → Results<br/>20-100ms"]

    Hybrid_Path["Hybrid Path<br/>10K < candidates < 100K<br/>VectorDB + GPU Rerank<br/>15-50ms"]

    Results["Deliver Results"]

    Query --> Parse
    Parse --> Estimate
    Estimate --> Analysis
    Analysis --> Decision

    Decision -->|GPU OK| GPU_Path
    Decision -->|Too Large| VDB_Path
    Decision -->|Medium| Hybrid_Path

    GPU_Path --> Results
    VDB_Path --> Results
    Hybrid_Path --> Results

    style Query fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style Decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style GPU_Path fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style VDB_Path fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style Hybrid_Path fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Results fill:#ffccbc,stroke:#d84315,stroke-width:2px
```"""

def convert_query_phases():
    """Convert ASCII query phases to Mermaid."""
    return """```mermaid
sequenceDiagram
    actor User
    participant Gateway as API Gateway
    participant Router as Query Router
    participant Embedding as Embedding Gen
    participant GPU as GPU Engine
    participant KG as Knowledge Graph
    participant RL as RL Personalization

    User->>Gateway: "French documentary climate"
    activate Gateway
    Gateway->>Gateway: Auth & Validation<br/>1-2ms

    Gateway->>Router: Parsed Query
    deactivate Gateway
    activate Router

    Router->>Embedding: Query Text
    deactivate Router
    activate Embedding
    Embedding->>Embedding: Generate 1024-dim<br/>Embedding 2-5ms

    Embedding->>Router: Query Vector
    deactivate Embedding
    activate Router

    Router->>Router: Complexity Analysis<br/>& Path Selection 0.1ms

    Router->>GPU: Load Filtered Candidates
    deactivate Router
    activate GPU
    GPU->>GPU: Tensor Core Similarity<br/>SSSP Engine 8-12ms

    GPU->>KG: Top-K Results
    deactivate GPU
    activate KG
    KG->>KG: Semantic Enrichment<br/>Graph Traversal 3-7ms

    KG->>RL: Enriched Results
    deactivate KG
    activate RL
    RL->>RL: Thompson Sampling<br/>Personalization 2-5ms

    RL->>User: Final Results<br/>JSON Serialization 0.5ms
    deactivate RL

    Note over User: Total Latency: ~15ms (p95)
```"""

# Main conversion function
def process_files():
    """Process markdown files and convert ASCII diagrams."""
    files_to_process = [
        '/home/devuser/workspace/hackathon-tv5/ARCHITECTURE.md',
        '/home/devuser/workspace/hackathon-tv5/design/architecture/system-architecture.md',
    ]

    conversions = {
        'system_context': convert_system_context_diagram(),
        'deployment_topology': convert_deployment_topology(),
        'multi_region': convert_multi_region_deployment(),
        'query_routing': convert_query_routing(),
        'query_phases': convert_query_phases(),
    }

    print("ASCII to Mermaid Conversion Results:")
    print("=" * 60)
    for name, content in conversions.items():
        print(f"\n{name.upper()}")
        print("-" * 60)
        print(content[:200] + "...")

    print("\n\nConversion complete. Use the above diagrams to replace broken ASCII diagrams in markdown files.")

if __name__ == '__main__':
    process_files()
