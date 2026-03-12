/**
 * benchmarks/retrieval_bench.js
 *
 * Measures retrieval latency at various corpus sizes.
 * Reports P50/P95/P99 latencies and throughput (QPS).
 *
 * Usage: node benchmarks/retrieval_bench.js
 */

import { HybridRetriever } from "../src/ranker/index.js";
import { rerank } from "../src/rerank/index.js";

const CORPUS_SIZES = [100, 500, 1000];
const QUERIES = [
  "neural information retrieval transformer",
  "BM25 lexical matching sparse",
  "dense embeddings similarity search",
  "cross-encoder reranking relevance",
  "retrieval augmented generation RAG",
];

function makeDoc(i) {
  const topics = [
    "Neural networks use backpropagation to learn weights from labeled data.",
    "Information retrieval systems rank documents by relevance to a query.",
    "Transformer models apply self-attention across token sequences.",
    "Vector databases store embeddings for approximate nearest neighbor search.",
    "Language models generate text by predicting next token probabilities.",
    "Semantic search finds contextually relevant results beyond keyword matching.",
    "Sparse retrieval methods like BM25 use inverted indexes for fast lookup.",
    "Dense retrieval uses dual encoders to embed queries and passages jointly.",
  ];
  return {
    id: `doc-${i}`,
    text: topics[i % topics.length] + ` (document ${i})`,
  };
}

function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.floor((p / 100) * sorted.length)];
}

async function benchRetriever(corpusSize) {
  const docs = Array.from({ length: corpusSize }, (_, i) => makeDoc(i));
  const retriever = new HybridRetriever(docs);

  const latencies = [];
  for (const q of QUERIES) {
    const t0 = performance.now();
    await retriever.retrieve(q, 10);
    latencies.push(performance.now() - t0);
  }

  return {
    corpusSize,
    p50: percentile(latencies, 50).toFixed(1),
    p95: percentile(latencies, 95).toFixed(1),
    p99: percentile(latencies, 99).toFixed(1),
    avg: (latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(1),
  };
}

console.log("pplx-retrieval-kernel benchmark\n");
console.log("Corpus Size | P50 (ms) | P95 (ms) | P99 (ms) | Avg (ms)");
console.log("─".repeat(60));

for (const size of CORPUS_SIZES) {
  const r = await benchRetriever(size);
  console.log(
    `${String(r.corpusSize).padEnd(11)} | ${r.p50.padEnd(8)} | ${r.p95.padEnd(8)} | ${r.p99.padEnd(8)} | ${r.avg}`
  );
}

console.log("\n✓ Benchmark complete");
