# Contributing to RustML

Thanks for your interest in contributing! Here's how to get started.

## Ways to Contribute

### 🟢 Easy (Good First Issues)
- Port a single `scipy.stats` function (e.g., `ttest_ind`, `pearsonr`, `norm.pdf`)
- Add a Python vs Rust benchmark for a common operation
- Improve docs or add examples
- Fix typos or improve error messages

### 🟡 Medium
- Port a scikit-learn algorithm (e.g., KMeans, PCA, StandardScaler)
- Implement a `rustml-chain` LLM provider (OpenAI, Anthropic, Ollama)
- Add PyO3 Python bindings for an existing crate
- Write integration tests

### 🔴 Advanced
- Implement a transformers pipeline (text-classification, embeddings)
- Port a gradient boosting tree implementation
- Optimize SIMD/vectorized operations
- Add GPU support via wgpu

## Development Setup

```bash
# Clone
git clone https://github.com/kikerub/rustml.git
cd rustml

# Build
cargo build

# Test
cargo test

# Bench
cargo bench
```

## Code Standards

- **Tests required**: Every function needs unit tests
- **Benchmarks encouraged**: Include Python comparison when possible
- **Documentation**: Public functions need doc comments with examples
- **Naming**: Follow Rust conventions, but keep Python-familiar names where sensible
- **No unsafe**: Unless absolutely necessary with clear justification

## Pull Request Process

1. Fork the repo
2. Create a feature branch (`feat/stats-ttest`)
3. Write code + tests + docs
4. Run `cargo test && cargo clippy && cargo fmt`
5. Open PR with description of what and why

## Architecture Decisions

- Each library is a separate crate in `crates/`
- Workspace-level `Cargo.toml` manages dependencies
- `ndarray` is the standard tensor type (don't reinvent)
- `serde` for serialization
- `rayon` for CPU parallelism
- `pyo3` for Python bindings
