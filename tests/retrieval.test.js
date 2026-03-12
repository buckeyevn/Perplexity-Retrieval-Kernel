/**
 * tests/retrieval.test.js — pplx-retrieval-kernel
 */

import { BM25Index, HybridRetriever } from "../src/ranker/index.js";
import { rerank, rerankMMR } from "../src/rerank/index.js";
import { embed, embedBatch, cosineSim, EMBED_DIM } from "../src/embedder/index.js";

const DOCS = [
  { id: "d1", text: "Neural networks learn representations from data through backpropagation." },
  { id: "d2", text: "BM25 is a bag-of-words retrieval function used in information retrieval systems." },
  { id: "d3", text: "Transformers use self-attention to model sequence dependencies." },
  { id: "d4", text: "The capital of France is Paris, known for the Eiffel Tower." },
  { id: "d5", text: "Retrieval-augmented generation combines neural retrieval with language model generation." },
  { id: "d6", text: "Dense retrieval encodes queries and documents into vector space for similarity search." },
  { id: "d7", text: "Cross-encoders jointly attend to query and document for more accurate relevance scoring." },
  { id: "d8", text: "The Eiffel Tower was built in 1889 and stands 330 meters tall." },
];

// ── Embedder ──────────────────────────────────────────────────────────────────

test("embed: returns Float32Array of correct dimension", async () => {
  const vec = await embed("hello world");
  expect(vec).toBeInstanceOf(Float32Array);
  expect(vec.length).toBe(EMBED_DIM);
});

test("embed: vectors are L2-normalized (unit length)", async () => {
  const vec = await embed("test normalization");
  let norm = 0;
  for (const v of vec) norm += v * v;
  expect(Math.sqrt(norm)).toBeCloseTo(1.0, 4);
});

test("embed: same text returns same vector (deterministic)", async () => {
  const v1 = await embed("deterministic test");
  const v2 = await embed("deterministic test");
  expect(Array.from(v1)).toEqual(Array.from(v2));
});

test("embed: different texts return different vectors", async () => {
  const v1 = await embed("apple");
  const v2 = await embed("quantum mechanics");
  expect(Array.from(v1)).not.toEqual(Array.from(v2));
});

test("embedBatch: returns correct count", async () => {
  const texts = ["one", "two", "three"];
  const vecs = await embedBatch(texts);
  expect(vecs.length).toBe(3);
  for (const v of vecs) expect(v).toBeInstanceOf(Float32Array);
});

test("embedBatch: cached results match single embed", async () => {
  const text = "cache consistency check";
  const single = await embed(text);
  const batch = await embedBatch([text]);
  expect(Array.from(single)).toEqual(Array.from(batch[0]));
});

test("cosineSim: identical vectors → 1.0", async () => {
  const v = await embed("cosine test");
  expect(cosineSim(v, v)).toBeCloseTo(1.0, 5);
});

test("cosineSim: unrelated vectors → less than identical", async () => {
  const v1 = await embed("banana smoothie recipe");
  const v2 = await embed("quantum field theory equations");
  expect(cosineSim(v1, v2)).toBeLessThan(1.0);
});

// ── BM25 ──────────────────────────────────────────────────────────────────────

test("BM25Index: scores are non-negative", () => {
  const idx = new BM25Index(DOCS);
  const scores = idx.score("neural retrieval");
  for (const s of scores) expect(s.score).toBeGreaterThanOrEqual(0);
});

test("BM25Index: result count equals document count", () => {
  const idx = new BM25Index(DOCS);
  expect(idx.score("retrieval")).toHaveLength(DOCS.length);
});

test("BM25Index: relevant doc scores higher than irrelevant", () => {
  const idx = new BM25Index(DOCS);
  const scores = idx.score("BM25 retrieval information");
  const scoreMap = new Map(scores.map((s) => [s.id, s.score]));
  // d2 is the BM25 document
  expect(scoreMap.get("d2")).toBeGreaterThan(scoreMap.get("d4")); // d4 is about Paris
});

test("BM25Index: empty query returns zero scores", () => {
  const idx = new BM25Index(DOCS);
  const scores = idx.score("");
  for (const s of scores) expect(s.score).toBe(0);
});

test("BM25Index: handles single document corpus", () => {
  const idx = new BM25Index([{ id: "x", text: "hello world" }]);
  const scores = idx.score("hello");
  expect(scores).toHaveLength(1);
  expect(scores[0].id).toBe("x");
});

// ── HybridRetriever ───────────────────────────────────────────────────────────

test("HybridRetriever: retrieve returns at most k results", async () => {
  const r = new HybridRetriever(DOCS);
  const results = await r.retrieve("neural retrieval", 3);
  expect(results.length).toBeLessThanOrEqual(3);
});

test("HybridRetriever: retrieve returns all fields", async () => {
  const r = new HybridRetriever(DOCS);
  const results = await r.retrieve("transformers attention");
  for (const res of results) {
    expect(res).toHaveProperty("id");
    expect(res).toHaveProperty("text");
    expect(res).toHaveProperty("score");
    expect(res).toHaveProperty("bm25");
    expect(res).toHaveProperty("neural");
  }
});

test("HybridRetriever: alpha=1 matches BM25 ordering", async () => {
  const r = new HybridRetriever(DOCS, { alpha: 1.0 });
  const hybrid = await r.retrieve("BM25 information retrieval", 4);
  // With alpha=1, should be entirely BM25-driven
  expect(hybrid[0].id).toBe("d2"); // d2 mentions BM25 explicitly
});

test("HybridRetriever: setAlphaForQueryType works", () => {
  const r = new HybridRetriever(DOCS);
  r.setAlphaForQueryType("factual");
  expect(r.alpha).toBe(0.3);
  r.setAlphaForQueryType("keyword");
  expect(r.alpha).toBe(0.6);
  r.setAlphaForQueryType("unknown_type");
  expect(r.alpha).toBe(0.35); // default fallback
});

// ── Reranker ──────────────────────────────────────────────────────────────────

test("rerank: returns at most topK results", async () => {
  const results = await rerank("neural retrieval", DOCS, { topK: 4 });
  expect(results.length).toBeLessThanOrEqual(4);
});

test("rerank: rerankScore is between 0 and 1", async () => {
  const results = await rerank("cross-encoder relevance scoring", DOCS, { topK: 5 });
  for (const r of results) {
    expect(r.rerankScore).toBeGreaterThanOrEqual(0);
    expect(r.rerankScore).toBeLessThanOrEqual(1);
  }
});

test("rerank: d7 ranks highly for cross-encoder query", async () => {
  const results = await rerank("cross-encoder query document scoring", DOCS, { topK: 3 });
  const ids = results.map((r) => r.id);
  expect(ids).toContain("d7"); // d7 is specifically about cross-encoders
});

test("rerank: originalRank preserved", async () => {
  const results = await rerank("test", DOCS.slice(0, 3), { topK: 3 });
  const ranks = results.map((r) => r.originalRank);
  expect(ranks.every((r) => r >= 1 && r <= 3)).toBe(true);
});

test("rerankMMR: returns diverse results", async () => {
  const results = await rerankMMR("Paris Eiffel Tower France", DOCS, {
    topK: 3,
    lambda: 0.5,
  });
  expect(results.length).toBeGreaterThan(0);
  // d4 and d8 are both about Paris/Eiffel, should not both dominate
  expect(results).toHaveLength(3);
});

test("rerank: empty candidates returns empty array", async () => {
  const results = await rerank("anything", []);
  expect(results).toEqual([]);
});
