# Semantic Model - ONNX Runtime Backend

High-performance sentence transformer implementation using ONNX Runtime for optimized inference.

## Features

- **ONNX Runtime Backend**: Leverages ONNX Runtime 2.0 for fast inference
- **GPU Acceleration**: CUDA support for GPU-accelerated inference
- **Efficient Tokenization**: Hugging Face tokenizers integration
- **Multiple Pooling Strategies**: Mean, Max, and CLS pooling
- **Batch Processing**: Efficient batch encoding support
- **Thread-Safe**: Interior mutability for concurrent inference

## Performance

Target performance metrics:
- **Inference Speed**: <5ms per query (vs 11ms Python baseline)
- **Throughput**: High queries-per-second with batch processing
- **Memory Efficient**: Optimized tensor operations

## Usage

### Basic Usage

```rust
use semantic_model::SemanticModel;

// Load model
let model = SemanticModel::new(
    "models/all-MiniLM-L6-v2.onnx",
    "models/tokenizer.json"
)?;

// Encode single text
let embedding = model.encode("Hello, world!")?;
assert_eq!(embedding.len(), 384);

// Encode batch
let texts = vec!["First text".to_string(), "Second text".to_string()];
let embeddings = model.encode_batch(&texts)?;
```

### Custom Configuration

```rust
use semantic_model::{ModelConfig, PoolingStrategy, SemanticModel};

let config = ModelConfig {
    max_length: 512,
    normalize: true,
    pooling_strategy: PoolingStrategy::Mean,
    use_gpu: true,
    embedding_dim: 384,
};

let model = SemanticModel::with_config(
    "models/model.onnx",
    "models/tokenizer.json",
    config,
)?;
```

### Semantic Search

```rust
use semantic_model::{SemanticModel, cosine_similarity};

let model = SemanticModel::new("model.onnx", "tokenizer.json")?;

let query = "machine learning";
let documents = vec![
    "AI and ML are transforming industries".to_string(),
    "Cooking recipes for beginners".to_string(),
];

let query_emb = model.encode(query)?;
let doc_embs = model.encode_batch(&documents)?;

for (doc, emb) in documents.iter().zip(doc_embs.iter()) {
    let similarity = cosine_similarity(&query_emb, emb);
    println!("{}: {:.4}", doc, similarity);
}
```

## Architecture

### Components

1. **SemanticModel**: Main interface for encoding
2. **ONNX Session**: ONNX Runtime session with GPU support
3. **Tokenizer**: Hugging Face tokenizer for text preprocessing
4. **Pooling**: Mean/Max/CLS pooling strategies
5. **Preprocessing**: Text normalization and cleaning

### Thread Safety

The model uses `Arc<Mutex<Session>>` for thread-safe inference:

```rust
// Safe to use across threads
let model = Arc::new(SemanticModel::new(...)?);

let handles: Vec<_> = (0..4)
    .map(|_| {
        let model = Arc::clone(&model);
        thread::spawn(move || {
            model.encode("test").unwrap()
        })
    })
    .collect();
```

## Configuration Options

### ModelConfig

- `max_length`: Maximum sequence length (default: 512)
- `normalize`: Normalize embeddings to unit length (default: true)
- `pooling_strategy`: Pooling method (default: Mean)
- `use_gpu`: Enable CUDA if available (default: true)
- `embedding_dim`: Output dimension (default: 384)

### Pooling Strategies

- **Mean**: Average token embeddings (weighted by attention mask)
- **Max**: Element-wise maximum across tokens
- **CLS**: Use only [CLS] token embedding

## ONNX Model Export

To use this crate, you need an ONNX-exported sentence transformer model:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"

# Export model
model = ORTModelForFeatureExtraction.from_pretrained(
    model_id,
    export=True
)
model.save_pretrained("models/")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("models/")
```

## Dependencies

- `ort` - ONNX Runtime bindings
- `tokenizers` - HuggingFace tokenizers
- `ndarray` - Array operations
- `parking_lot` - Efficient synchronization primitives

## Features

- `default` = `["onnx", "cuda"]`
- `onnx` - Enable ONNX Runtime support
- `cuda` - Enable CUDA acceleration
- `cpu-only` - CPU-only inference

## Examples

See `examples/basic_usage.rs` for a comprehensive example:

```bash
cargo run --example basic_usage
```

## Benchmarks

Run benchmarks:

```bash
cargo bench -p semantic-model
```

Benchmark categories:
- Single text encoding
- Batch encoding (various sizes)
- Pooling strategies comparison
- Throughput measurement

## Testing

```bash
# Unit tests
cargo test -p semantic-model

# Integration tests (requires model files)
cargo test -p semantic-model --test integration_tests -- --ignored

# All tests
cargo test -p semantic-model --all-targets
```

## Performance Tips

1. **Use Batch Encoding**: Process multiple texts together for better throughput
2. **Enable GPU**: Use CUDA for significant speedup on GPU-enabled systems
3. **Warm-up**: Run a few inferences before measuring performance
4. **Reuse Model**: Create model once and reuse across requests

## Compatibility

- Rust: 1.75+
- ONNX Runtime: 2.0+
- CUDA: 11.0+ (optional, for GPU support)

## License

MIT
