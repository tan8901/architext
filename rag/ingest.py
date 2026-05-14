"""
rag/ingest.py — Knowledge Base Ingestion

Loads tech documentation and architecture patterns into ChromaDB.
Run this ONCE before starting the agent:

    python -m rag.ingest

The knowledge base is seeded with curated content covering:
- Common architecture patterns with their trade-offs
- Cloud service cost reference data
- Compliance requirements (HIPAA, GDPR, SOC2)
- Scaling strategies and decision heuristics

You can extend this by pointing it at your own docs (PDFs, markdown files).
"""

from __future__ import annotations
import os
import logging
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

COLLECTION_NAME = "architext_knowledge"
CHROMA_DIR = "./data/chroma_db"

# ── Curated knowledge base ──────────────────────────────────────────────────
# Each entry: (category, source_label, content)
# In a production system you'd scrape/load these from real docs.
# This seed set is enough to meaningfully improve generation quality.

KNOWLEDGE_BASE = [
    # ── Architecture Patterns ──
    ("pattern", "monolith-first",
     """Monolith-First Pattern: Start with a single deployable unit. 
     Best for teams under 10 engineers and <50k monthly users. 
     Pros: simpler deployment, easier debugging, lower infrastructure cost ($50-200/mo).
     Cons: scaling bottlenecks above 100k users, technology lock-in.
     Decision rule: Use monolith-first when time-to-market < 3 months or team < 6 engineers."""),

    ("pattern", "microservices",
     """Microservices Architecture: Decompose system into independently deployable services.
     Best for teams of 10+ engineers and >100k users.
     Each service owns its data store (database per service pattern).
     Requires: API gateway, service discovery, distributed tracing, container orchestration.
     Cost: 3-5x higher infrastructure cost than equivalent monolith due to overhead services.
     Anti-pattern: microservices with small teams causes distributed monolith — worst of both worlds."""),

    ("pattern", "event-driven",
     """Event-Driven Architecture: Services communicate via events/messages rather than direct calls.
     Core components: Message broker (Kafka, RabbitMQ, AWS SQS), event producers, event consumers.
     Best for: real-time processing, audit trails, decoupling services.
     Kafka handles millions of events/second; RabbitMQ better for task queues.
     Complexity cost: requires idempotency, event schema evolution strategy, dead letter queues."""),

    ("pattern", "serverless",
     """Serverless Architecture: Stateless functions triggered by events, auto-scaling to zero.
     AWS Lambda: $0.20 per 1M requests + compute time. Very cost-effective at low/variable traffic.
     Cold start latency: 100-500ms for Python/Node, up to 1s for JVM.
     Best for: event processing, scheduled jobs, APIs with unpredictable traffic.
     Anti-pattern: serverless for stateful, long-running, or latency-sensitive workloads."""),

    ("pattern", "cqrs",
     """CQRS (Command Query Responsibility Segregation):
     Separate read and write models. Writes go to command model, reads from optimized query model.
     Often paired with Event Sourcing: store events instead of current state.
     Use when: complex domain logic, different read/write scaling needs, audit requirements.
     Adds complexity: eventual consistency, projection maintenance, debugging difficulty.
     Best for enterprise applications, financial systems, healthcare records."""),

    ("pattern", "strangler-fig",
     """Strangler Fig Pattern for migrations: Gradually replace legacy system by routing traffic.
     Step 1: Put new system behind feature flags / proxy.
     Step 2: Route subset of traffic to new system.
     Step 3: Gradually increase traffic as confidence grows.
     Step 4: Decommission legacy once 100% migrated.
     Minimizes risk: zero downtime migration, easy rollback."""),

    # ── Technology Reference ──
    ("tech", "postgresql",
     """PostgreSQL: Best relational database for most applications.
     Handles 10k-100k queries/second on modern hardware.
     Managed options: AWS RDS ($25-300/mo), Google Cloud SQL ($25-300/mo), Supabase (free tier).
     Use PostgreSQL for: transactional data, complex queries, ACID compliance, JSON data (JSONB).
     Scale-out: read replicas for read-heavy workloads, connection pooling via PgBouncer."""),

    ("tech", "redis",
     """Redis: In-memory data structure store, primary use case is caching.
     Sub-millisecond latency for cache hits. 
     AWS ElastiCache Redis: $15-200/mo depending on instance size.
     Use cases: session storage, rate limiting, pub/sub for real-time features, leaderboards.
     Redis Cluster: horizontal scaling for >100GB data or >100k ops/second."""),

    ("tech", "api-gateway",
     """API Gateway options:
     AWS API Gateway: $3.50 per million API calls. Handles auth, rate limiting, SSL.
     Kong: Open source, self-hosted. Feature-rich but operational overhead.
     Nginx: Simple reverse proxy, free. Good for basic load balancing.
     For microservices: API gateway is mandatory for external traffic routing.
     BFF (Backend for Frontend): separate API gateways per client type (mobile, web)."""),

    ("tech", "message-queues",
     """Message Queue Comparison:
     AWS SQS: $0.40 per million messages. Fully managed, simple. Best for task queues.
     RabbitMQ: Open source, complex routing, lower latency. Self-hosted: $50-200/mo.
     Apache Kafka: Best for high-throughput event streaming (>100k events/sec). 
     AWS MSK (Kafka): $200-500/mo minimum.
     Choose SQS for simple job queues, Kafka for event streaming, RabbitMQ for complex routing."""),

    ("tech", "websockets",
     """WebSocket/Real-time Options:
     AWS API Gateway WebSocket: $1 per million messages + connection time.
     Socket.io: Self-hosted, requires sticky sessions or Redis adapter for multi-node.
     Pusher: Managed WebSockets, $49/mo for 500 concurrent connections.
     Firebase Realtime DB: Managed, $25/mo for 100GB storage + bandwidth.
     For >10k concurrent connections: use Redis pub/sub with WebSocket server clusters."""),

    # ── Compliance ──
    ("compliance", "hipaa",
     """HIPAA Technical Safeguards Requirements:
     1. Access Controls: unique user IDs, emergency access procedure, automatic logoff.
     2. Audit Controls: record and examine hardware/software activity.
     3. Integrity Controls: protect ePHI from alteration or destruction.
     4. Transmission Security: encrypt ePHI over open networks (TLS 1.2+).
     Required: Business Associate Agreements (BAA) with all cloud vendors.
     AWS HIPAA eligible services: EC2, RDS, S3, Lambda, and 100+ others.
     Architecture must include: encryption at rest, encryption in transit, audit logs, MFA."""),

    ("compliance", "gdpr",
     """GDPR Technical Requirements:
     1. Data minimization: only collect what you need.
     2. Right to erasure: must be able to delete all user data within 30 days.
     3. Data portability: export user data in machine-readable format.
     4. Data residency: EU user data must stay in EU regions.
     5. Breach notification: notify within 72 hours of discovery.
     Architecture implications: soft delete pattern, data export API, EU-only deployments."""),

    ("compliance", "soc2",
     """SOC 2 Type II Requirements (Security Trust Service Criteria):
     Availability: 99.9%+ uptime SLA.
     Confidentiality: encrypt sensitive data, access controls, least privilege.
     Processing Integrity: monitoring, alerting, incident response.
     Privacy: data handling policies.
     Required architecture: centralized logging, anomaly detection, change management."""),

    # ── Scaling Heuristics ──
    ("scaling", "user-scale-guide",
     """User Scale Architecture Decision Guide:
     0-1k users: Single server, SQLite or Postgres. $20-50/mo.
     1k-10k users: Small VPS + managed Postgres. Basic caching. $50-200/mo.
     10k-100k users: Load balanced app servers + managed DB + Redis cache. $200-1000/mo.
     100k-1M users: Auto-scaling groups + read replicas + CDN + async jobs. $1000-5000/mo.
     1M+ users: Microservices + sharded DBs + multi-region. $5000+/mo.
     Rule of thumb: budget 10x headroom — systems rarely stay at current scale."""),

    ("scaling", "database-scaling",
     """Database Scaling Patterns:
     Read replicas: add read replicas for read-heavy workloads (>80% reads).
     Connection pooling: PgBouncer reduces connections; critical at >100 concurrent users.
     Caching layer: Redis cache for hot data reduces DB load by 80-90%.
     Vertical scaling: increase instance size before horizontal; simpler, cheap to $500/mo.
     Sharding: only needed at 100M+ records or >10k writes/second. High complexity cost.
     CQRS: separate read and write models for complex read/write patterns."""),

    # ── Cost Estimation ──
    ("cost", "aws-pricing-2024",
     """AWS Cost Reference (2024):
     EC2 t3.micro: $8/mo. t3.small: $17/mo. t3.medium: $33/mo. t3.large: $67/mo.
     RDS PostgreSQL db.t3.micro: $25/mo. db.t3.small: $50/mo. db.t3.medium: $98/mo.
     ElastiCache Redis cache.t3.micro: $14/mo. cache.t3.small: $27/mo.
     S3: $0.023/GB/mo storage + $0.09/GB egress.
     CloudFront CDN: $0.01/GB for first 10TB/mo.
     Application Load Balancer: $16/mo + $0.008 per LCU hour.
     SQS: $0.40 per million requests.
     Lambda: $0.20 per million requests + $0.0000166667 per GB-second."""),
]


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_knowledge_base(
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    force_reload: bool = False,
) -> int:
    """
    Ingest the curated knowledge base into ChromaDB.
    
    Returns number of documents ingested.
    """
    logger.info(f"📚 Ingesting knowledge base into {chroma_dir}...")

    client = chromadb.PersistentClient(path=chroma_dir)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # Check if collection exists and has data
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections and not force_reload:
        collection = client.get_collection(collection_name, embedding_function=embedding_fn)
        count = collection.count()
        if count > 0:
            logger.info(f"✅ Collection '{collection_name}' already has {count} documents. Skipping.")
            return count

    # Create or recreate collection
    if collection_name in existing_collections:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents = []
    metadatas = []
    ids = []

    for category, source, content in KNOWLEDGE_BASE:
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{source}_{i}".encode()).hexdigest()[:12]
            documents.append(chunk)
            metadatas.append({"category": category, "source": source, "chunk": i})
            ids.append(doc_id)

    # Also load any .md or .txt files from data/docs/
    docs_path = Path("./data/docs")
    if docs_path.exists():
        for filepath in docs_path.glob("**/*.{md,txt}"):
            try:
                text = filepath.read_text(encoding="utf-8")
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    doc_id = hashlib.md5(f"{filepath.name}_{i}".encode()).hexdigest()[:12]
                    documents.append(chunk)
                    metadatas.append({
                        "category": "custom",
                        "source": filepath.stem,
                        "chunk": i,
                    })
                    ids.append(doc_id)
                logger.info(f"   Loaded: {filepath.name} ({len(chunks)} chunks)")
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

    # Batch upsert (ChromaDB handles duplicates)
    BATCH_SIZE = 100
    for i in range(0, len(documents), BATCH_SIZE):
        collection.upsert(
            documents=documents[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE],
        )

    total = collection.count()
    logger.info(f"✅ Ingested {total} document chunks into '{collection_name}'")
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ingest_knowledge_base(force_reload=True)
