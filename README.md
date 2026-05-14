# 🏗️ Architext — AI Solutions Architect Agent

> Transform product requirements into production-ready architecture decisions in ~90 seconds.

**Built with:** LangGraph · LangChain · Ollama (Llama 3.2) · ChromaDB · Streamlit

<img width="1918" height="586" alt="image" src="https://github.com/user-attachments/assets/9711b4bd-3e92-4d84-b23b-4b52d38f3c76" />

---

## What it does

Architext is a stateful AI agent that reads your project requirements and produces:

- **3 architecture options** with detailed trade-off analysis
- **Scored comparison matrix** across scalability, cost, complexity, and maintainability
- **Mermaid architecture diagrams** for each approach
- **Cloud cost estimates** (~±20% accuracy)
- **Risk assessment** (compliance gaps, scaling risks, team capacity)
- **Prioritized implementation roadmap**

### Example
> Demonstration Video is available in the repository as Architext_Demo.mp4
<img width="1918" height="868" alt="image" src="https://github.com/user-attachments/assets/258b509c-d9d1-4640-9121-7eac3b1b3077" />
<img width="1918" height="732" alt="image" src="https://github.com/user-attachments/assets/b426642e-125e-45d5-a897-e8e2ce45aabf" />
<img width="1918" height="867" alt="image" src="https://github.com/user-attachments/assets/4fa6fa57-8e49-4562-afb1-b732cf585f84" />

---

## Architecture

```
User Input (text / PRD doc)
         │
┌────────▼──────────────────────────────────────┐
│              LangGraph Workflow                │
│                                                │
│  1. extract_requirements                       │
│     └── Parses reqs, constraints, scale        │
│         └── (needs_clarification?) → END       │
│                                                │
│  2. retrieve_context                           │
│     └── Queries ChromaDB RAG knowledge base   │
│         (arch patterns, cost data, compliance) │
│                                                │
│  3. generate_approaches                        │
│     └── Generates 3 distinct architectures    │
│         with components, costs, diagrams       │
│                                                │
│  4. synthesize_report                          │
│     └── Ranks approaches, builds comparison   │
│         table, infers risks, writes next steps │
└────────────────────────────────────────────────┘
         │
   Final Report (Markdown + JSON)
```

**Key design decisions:**
- **Stateful graph** (LangGraph) — not a simple prompt chain. State is checkpointed between nodes, enabling resumability and conditional branching.
- **RAG over prompting alone** — ChromaDB stores curated tech docs, pricing data, and compliance requirements. This grounds the LLM and reduces hallucination on specifics like cloud costs.
- **Separation of concerns** — deterministic logic (comparison table, risk scoring) is computed in Python, not delegated to the LLM. LLM is only used where natural language quality matters.
- **Partial application for dependency injection** — nodes are pure functions tested in isolation; shared resources (LLM, ChromaDB client) are bound via `functools.partial`.

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running

### Install

```bash
# 1. Clone and install dependencies
git clone https://github.com/yourusername/architext.git
cd architext
pip install -r requirements.txt

# 2. Pull the model (one-time, ~2GB)
ollama pull llama3.2

# 3. Copy config
cp .env.example .env

# 4. Initialize the knowledge base (one-time)
python run.py --ingest
```

### Run

**Option A: Streamlit UI** (recommended for demos)
```bash
streamlit run ui/app.py
```

**Option B: CLI** (faster for testing)
```bash
python run.py "Build a healthcare patient portal, HIPAA compliant, 2000 users, team of 6"
```

**Option C: Python API**
```python
from agent.graph import run_agent

state = run_agent("Build a real-time analytics dashboard for a SaaS product")
report = state["report"]

print(report["recommended"]["name"])          # "Modular Monolith"
print(report["recommended"]["total_monthly_cost_usd"])  # 320.0
print(report["comparison_table"])             # markdown table
```

---

## Project Structure

```
architext/
├── agent/
│   ├── state.py          # TypedDict state schema
│   ├── graph.py          # LangGraph workflow assembly
│   └── nodes/
│       ├── extractor.py  # Requirement parsing node
│       ├── retriever.py  # RAG retrieval node
│       ├── generator.py  # Architecture generation node
│       └── synthesizer.py# Report compilation node
├── rag/
│   └── ingest.py         # Knowledge base ingestion
├── ui/
│   └── app.py            # Streamlit frontend
├── tests/
│   └── test_agent.py     # Unit tests (mock LLM)
├── data/
│   └── docs/             # Drop custom .md/.txt docs here
├── run.py                # CLI entrypoint
└── requirements.txt
```

---

## Running Tests

```bash
# Unit tests (no Ollama needed — LLM is mocked)
pytest tests/ -v

# Integration smoke test (requires Ollama running)
pytest tests/ -v -k smoke
```

---

## Extending the Knowledge Base

Drop any `.md` or `.txt` files into `data/docs/` and re-run ingestion:

```bash
python run.py --ingest
```

Good candidates:
- Your company's preferred tech stack docs
- AWS/GCP/Azure service documentation
- Industry-specific compliance requirements
- Internal architecture decision records (ADRs)

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model to use |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector DB storage path |
| `TAVILY_API_KEY` | *(optional)* | For web search (latest pricing data) |

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Llama 3.2 via Ollama | Local, free, private |
| Orchestration | LangGraph 0.2 | Stateful workflows, conditional edges |
| Vector DB | ChromaDB | Local, zero-config, fast enough for this scale |
| Embeddings | ChromaDB default (all-MiniLM) | No extra setup |
| Frontend | Streamlit | Fastest path to a working demo UI |
| Testing | pytest + unittest.mock | Mock LLM for fast, offline tests |

---

## Limitations & Future Work

- **Cost estimates are approximate** (~±30% on Llama 3.2 vs ±20% on GPT-4). Accuracy improves with Tavily web search enabled.
- **Diagram quality varies** — Mermaid syntax is finicky; some diagrams may need manual cleanup.
- **No memory between sessions** — add LangGraph's `SqliteSaver` checkpointer for persistence.
- **No streaming UI** — Streamlit shows progress by polling a background thread. A FastAPI + WebSocket backend would give real-time streaming.

---

## License

MIT
