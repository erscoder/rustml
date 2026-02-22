# 🦀 RustML — Python ML Libraries, Rewritten in Rust

> Python had its rise because of its learning curve when code was 100% human-written. With RAM as the bottleneck for local AI and servers — and memory becoming the most expensive component — Rust will have its ATH in the coming years.

**The mission:** Port the most important Python ML/AI libraries to Rust, one by one. Open source. Community-driven.

## Why

| Factor | Python | Rust |
|--------|--------|------|
| Memory usage | 3-10x overhead (GC, objects) | Near-zero overhead |
| Inference speed | Slow (CPython), fast only via C bindings | Native speed, no FFI needed |
| Concurrency | GIL bottleneck | Fearless concurrency |
| Edge/Mobile | Barely viable | First-class target |
| Safety | Runtime crashes | Compile-time guarantees |
| Binary size | ~100MB+ with deps | ~5-15MB static binary |

**The real argument:** Every Python ML library that matters (NumPy, PyTorch, tokenizers) is actually C/C++/Rust under the hood. Python is just the glue. Why not cut out the middleman?

## The Gap — What Exists vs What's Missing

### ✅ Already Good in Rust
| Python Library | Rust Equivalent | Stars | Maturity |
|---------------|-----------------|-------|----------|
| pandas | **polars** | 32K⭐ | 🟢 Production |
| numpy (arrays) | **ndarray** | 3.5K⭐ | 🟢 Stable |
| PyTorch (inference) | **candle** (HuggingFace) | 16K⭐ | 🟡 Growing |
| PyTorch (training) | **burn** | 9K⭐ | 🟡 Growing |
| scikit-learn | **linfa** | 3.7K⭐ | 🟡 Partial |
| tokenizers | **tokenizers** (HF) | 9K⭐ | 🟢 Production |
| huggingface_hub | **hf-hub** | 500⭐ | 🟡 Basic |
| regex/text | **regex** | 5K⭐ | 🟢 Production |

### 🔴 Missing — Our Targets
| Python Library | What It Does | Priority | Est. Effort |
|---------------|-------------|----------|-------------|
| **scipy** | Scientific computing (linalg, optimize, signal, stats) | 🔥 P0 | Large |
| **transformers** | Model hub + pipelines (BERT, GPT, etc.) | 🔥 P0 | Large |
| **langchain** | LLM app framework (chains, agents, RAG) | 🔥 P0 | Medium |
| **matplotlib** | Plotting/visualization | P1 | Large |
| **scikit-learn** (full) | Missing: SVM, clustering, preprocessing, pipelines | P1 | Medium |
| **sentence-transformers** | Embeddings for semantic search | P1 | Medium |
| **ONNX Runtime** | Model inference (multi-framework) | P1 | Medium |
| **xgboost/lightgbm** | Gradient boosting | P1 | Medium |
| **ray** | Distributed computing | P2 | Large |
| **dask** | Parallel dataframes | P2 | Medium (polars covers some) |
| **spacy** | NLP pipelines (NER, POS, parsing) | P2 | Large |
| **opencv** | Computer vision | P2 | Large |
| **requests/httpx** | HTTP client (AI API calls) | ✅ reqwest exists | — |
| **pydantic** | Data validation | ✅ serde exists | — |
| **fastapi** | Web framework | ✅ axum/actix exist | — |

## Project Structure

```
rustml/
├── crates/
│   ├── rustml-scipy/        # Scientific computing primitives
│   │   ├── linalg/          # Linear algebra (beyond ndarray)
│   │   ├── optimize/        # Optimization algorithms
│   │   ├── signal/          # Signal processing
│   │   └── stats/           # Statistical functions
│   ├── rustml-transformers/  # Model hub + inference pipelines
│   │   ├── hub/             # Download models from HF
│   │   ├── pipelines/       # text-classification, embeddings, generation
│   │   └── models/          # Model implementations
│   ├── rustml-chain/         # LLM application framework
│   │   ├── llm/             # LLM providers (OpenAI, Anthropic, local)
│   │   ├── chain/           # Prompt chains
│   │   ├── rag/             # Retrieval-augmented generation
│   │   └── agents/          # Tool-using agents
│   ├── rustml-boost/         # Gradient boosting (XGBoost/LightGBM)
│   ├── rustml-viz/           # Plotting (SVG/PNG output)
│   └── rustml-embed/         # Sentence embeddings
├── benchmarks/               # Python vs Rust benchmarks
│   ├── memory/               # RAM usage comparisons
│   ├── speed/                # Execution time comparisons
│   └── results/              # Published benchmark results
├── examples/
│   ├── inference/            # Run a model locally
│   ├── rag-pipeline/         # Build a RAG system
│   ├── sentiment/            # Sentiment analysis
│   └── embeddings/           # Generate embeddings
├── python-bindings/          # PyO3 bindings (use from Python!)
├── BENCHMARKS.md
├── CONTRIBUTING.md
├── ROADMAP.md
└── Cargo.toml                # Workspace
```

## Roadmap

### Phase 1: Foundation + Benchmarks (Month 1-2)
- [ ] Workspace setup with CI/CD
- [ ] `rustml-scipy` — stats module (distributions, hypothesis tests, descriptive stats)
- [ ] `rustml-scipy` — linalg module (decompositions, solvers beyond ndarray)
- [ ] **Benchmarks repo**: Memory + speed comparisons Python vs Rust for 10 common operations
- [ ] README with compelling benchmark charts

### Phase 2: Transformers + LLM (Month 3-4)
- [ ] `rustml-transformers` — HF hub integration (download models, tokenizers)
- [ ] `rustml-transformers` — inference pipelines (text-classification, embeddings, generation)
- [ ] `rustml-chain` — LLM provider abstraction (OpenAI, Anthropic, Ollama)
- [ ] `rustml-chain` — basic chains + RAG pipeline

### Phase 3: ML Algorithms + Boost (Month 5-6)
- [ ] `rustml-boost` — XGBoost-compatible gradient boosting
- [ ] `rustml-embed` — sentence transformer inference
- [ ] Expand linfa contributions (SVM, clustering, full pipelines)

### Phase 4: Ecosystem + Community (Month 7+)
- [ ] `rustml-viz` — basic plotting (SVG charts, exportable)
- [ ] Python bindings via PyO3 (use Rust libs from Python seamlessly)
- [ ] `cargo install rustml` one-liner
- [ ] Conference talks + blog posts

## Benchmark Preview

```
┌────────────────────────────────┬──────────┬──────────┬──────────┐
│ Operation                      │ Python   │ Rust     │ Savings  │
├────────────────────────────────┼──────────┼──────────┼──────────┤
│ Load 1M row CSV                │ 850MB    │ 180MB    │ 79% RAM  │
│ Matrix multiply 4096x4096     │ 2.1s     │ 0.8s     │ 62% time │
│ Tokenize 100K sentences       │ 12.3s    │ 1.1s     │ 91% time │
│ BERT inference (batch 32)     │ 1.8s     │ 0.6s     │ 67% time │
│ LLM 7B inference (CPU)        │ 45 tok/s │ 120 tok/s│ 2.7x     │
│ Embedding 10K docs            │ 34s      │ 8s       │ 76% time │
│ Train XGBoost 100K rows       │ 8.2s     │ 3.1s     │ 62% time │
│ Idle memory (loaded model)    │ 4.2GB    │ 1.8GB    │ 57% RAM  │
└────────────────────────────────┴──────────┴──────────┴──────────┘
* Preliminary estimates based on candle/polars benchmarks. Full benchmarks TBD.
```

## Philosophy

1. **Benchmarks first** — Don't just port, prove it's better. Every crate ships with Python vs Rust benchmarks.
2. **API familiarity** — Name things like Python devs expect. `rustml.transformers.pipeline("sentiment-analysis")` should feel familiar.
3. **Incremental adoption** — PyO3 bindings let you use Rust crates from Python. Migrate one function at a time.
4. **Community > hero code** — Welcoming contributions, good docs, clear architecture.
5. **Ship small, ship often** — A working `stats` module beats a half-finished `scipy` clone.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Good first issues:**
- Port a scipy.stats function (e.g., `ttest_ind`, `pearsonr`)
- Add a benchmark for a common operation
- Improve documentation or examples
- Port a scikit-learn algorithm to linfa

## The Vision

In 3 years, a developer building an AI product should be able to:

```rust
use rustml_transformers::pipeline;
use rustml_chain::{Chain, LLM, RAG};
use polars::prelude::*;

// Load data (polars already exists)
let df = LazyFrame::scan_parquet("data.parquet")?;

// Run ML model
let classifier = pipeline("sentiment-analysis", "distilbert-base")?;
let results = classifier.predict(&texts)?;

// Build RAG pipeline
let rag = RAG::new()
    .embedder("all-MiniLM-L6-v2")
    .store(VectorStore::local("./index"))
    .llm(LLM::anthropic("claude-sonnet"))?;

let answer = rag.query("What are the key trends?")?;
```

All running in **~200MB RAM** instead of **~2GB**, as a **single static binary**, deployable anywhere.

---

## License

MIT OR Apache-2.0

## Star History

If you believe Rust is the future of ML infrastructure, give us a ⭐

---

*Born from a conversation about why Python's reign in ML is ending — not because Python is bad, but because hardware constraints demand better.*
