Please ensure the code fails immediately on any error. My dataset is binary (not multiclass).  Avoid unnecessary features like checkpointing or extra abstractions—focus strictly on functional code and bug fixes.


Next, create a doc/ directory and add a Markdown file containing a Mermaid diagram. The diagram should show a high-level overview, and within the same file (under the same chart name), include subgraphs for the lower-level details. This should be updated anytime the orifinal file is changed. Do not bloat this file. Look at this file before searching or reading the original file.



How access the model:
#### Single Text Generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "The future of artificial intelligence is",
    "max_new_tokens": 50,
    "temperature": 0.8,
    "top_k": 40
  }'
```

#### Batch Text Generation (Optimized)
For multiple claims - **2x faster with 8+ claims**:
```bash
curl -X POST http://localhost:8000/generate_batch \
  -H "Content-Type: application/json" \
  -d '{
    "claims": [
      "The future of AI is",
      "Machine learning will",
      "Deep learning enables"
    ],
    "max_new_tokens": 30,
    "temperature": 0.8
  }'
```

#### Streaming Generation
Real-time token-by-token generation:
```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Once upon a time",
    "max_new_tokens": 50,
    "temperature": 0.8,
    "stream": true
  }'
```

### Computing Embeddings

#### Single Request Embeddings
```bash
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "claims": [
      "Artificial intelligence will transform healthcare",
      "Climate change is a critical global issue",
      "Renewable energy is the future"
    ]
  }'
```

#### Batch Embeddings (Optimized)
For large datasets with custom batch size:
```bash
curl -X POST http://localhost:8000/embeddings_batch \
  -H "Content-Type: application/json" \
  -d '{
    "claims": [
      "AI enables predictive analytics",
      "Machine learning improves decision making",
      "Neural networks process complex patterns"
    ],
    "batch_size": 8
  }'
```

### Python Example

```python
import requests

# Text generation
response = requests.post("http://localhost:8000/generate", json={
    "claim": "The future of AI is",
    "max_new_tokens": 50,
    "temperature": 0.8
})
result = response.json()
print(result["generated_text"])

# Embeddings
response = requests.post("http://localhost:8000/embeddings", json={
    "claims": ["AI will revolutionize medicine", "Climate action is urgent"]
})
embeddings = response.json()["embeddings"]
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
```