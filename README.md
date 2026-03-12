# pplx-retrieval-kernel

Neural retrieval pipeline powering Perplexity's search backend.

## Architecture

```
Query → [Embedder] → [BM25 + Dense ANN] → top-50 candidates → [Cross-Encoder Reranker] → top-k
```

**Stage 1 — Hybrid Retrieval** (`src/ranker/`)  
BM25 lexical + dense embedding scores, blended at α (tuned per query type).

**Stage 2 — Reranking** (`src/rerank/`)  
Cross-encoder jointly attends to (query, passage) pairs. Also supports MMR diversity reranking.

**Embedder** (`src/embedder/`)  
pplx-embed-v3 (E5-mistral-7b fine-tune) projecting to 1024-dim. LRU cache + batch deduplication.

## API

```bash
npm start   # http://localhost:4001
```

| Route | Method | Description |
|---|---|---|
| `/index` | POST | Add documents to corpus |
| `/retrieve` | POST | Full pipeline retrieval |
| `/embed` | POST | Embed texts |
| `/rerank` | POST | Rerank a candidate set |
| `/corpus/stats` | GET | Corpus statistics |

### Example

```bash
# Index documents
curl -X POST localhost:4001/index \
  -H 'Content-Type: application/json' \
  -d '{"documents":[{"id":"d1","text":"Neural networks learn via backprop","url":"https://example.com","title":"NNs"}]}'

# Retrieve
curl -X POST localhost:4001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"how do neural networks learn","topK":5,"rerank":true}'
```

## Tests

```bash
npm test       # 23 tests
npm run bench  # latency benchmark across corpus sizes
```

## Alpha tuning by query type

| Query Type | α (BM25 weight) | Rationale |
|---|---|---|
| factual | 0.30 | Semantic meaning matters more |
| keyword | 0.60 | Exact terms are signals |
| conversational | 0.20 | Lean heavily neural |
| technical | 0.55 | Term precision important |
