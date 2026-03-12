/**
 * api/server.js — pplx-retrieval-kernel
 *
 * Routes:
 *   POST /retrieve       — full pipeline: embed + BM25 + neural + rerank
 *   POST /embed          — embed texts
 *   POST /rerank         — rerank a candidate set
 *   POST /index          — add documents to in-memory corpus
 *   GET  /corpus/stats   — corpus statistics
 *   GET  /health
 */

import express from "express";
import cors from "cors";
import { HybridRetriever } from "../ranker/index.js";
import { rerank, rerankMMR } from "../rerank/index.js";
import { embedBatch } from "../embedder/index.js";

const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" }));

/** In-memory document store */
let corpus = [];
let retriever = null;

function getRetriever() {
  if (!retriever || retriever._stale) {
    retriever = new HybridRetriever(corpus);
    if (retriever) retriever._stale = false;
  }
  return retriever;
}

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/health", (_, res) => {
  res.json({ status: "ok", corpusSize: corpus.length });
});

/**
 * POST /index
 * Add documents to the corpus.
 * Body: { documents: [{id, text, url?, title?, metadata?}] }
 */
app.post("/index", (req, res) => {
  const { documents } = req.body ?? {};
  if (!Array.isArray(documents) || !documents.length) {
    return res.status(400).json({ error: "documents[] required" });
  }

  let added = 0;
  const existingIds = new Set(corpus.map((d) => d.id));
  for (const doc of documents) {
    if (!doc.id || !doc.text) continue;
    if (existingIds.has(doc.id)) {
      // Update
      const i = corpus.findIndex((d) => d.id === doc.id);
      corpus[i] = doc;
    } else {
      corpus.push(doc);
      added++;
    }
  }

  // Invalidate retriever cache
  if (retriever) retriever._stale = true;

  res.json({ added, total: corpus.length });
});

/**
 * POST /retrieve
 * Full pipeline retrieve.
 * Body: {
 *   query: string,
 *   topK?: number,           (default 10)
 *   candidateK?: number,     (default 50)
 *   alpha?: number,          BM25 weight
 *   rerank?: boolean,        (default true)
 *   diversity?: boolean,     use MMR (default false)
 *   queryType?: string       factual|keyword|conversational|technical
 * }
 */
app.post("/retrieve", async (req, res) => {
  const {
    query,
    topK = 10,
    candidateK = 50,
    alpha,
    rerank: doRerank = true,
    diversity = false,
    queryType,
  } = req.body ?? {};

  if (!query) return res.status(400).json({ error: "query required" });
  if (corpus.length === 0)
    return res.status(400).json({ error: "Corpus is empty. POST /index first." });

  const t0 = Date.now();
  const r = getRetriever();

  if (queryType) r.setAlphaForQueryType(queryType);
  if (alpha !== undefined) r.alpha = alpha;

  // Stage 1: hybrid retrieval
  const candidates = await r.retrieve(query, Math.min(candidateK, corpus.length));

  // Stage 2: rerank
  let results = candidates;
  if (doRerank && candidates.length > 0) {
    results = diversity
      ? await rerankMMR(query, candidates, { topK })
      : await rerank(query, candidates, { topK });
  } else {
    results = candidates.slice(0, topK);
  }

  res.json({
    query,
    results: results.map((r, rank) => ({
      rank: rank + 1,
      id: r.id,
      text: r.text,
      title: r.title,
      url: r.url,
      score: r.rerankScore ?? r.score,
      hybridScore: r.score,
      bm25: r.bm25,
      neural: r.neural,
    })),
    meta: {
      corpusSize: corpus.length,
      candidateK: candidates.length,
      topK: results.length,
      alpha: r.alpha,
      reranked: doRerank,
      latencyMs: Date.now() - t0,
    },
  });
});

/**
 * POST /embed
 * Body: { texts: string[], model?: string }
 */
app.post("/embed", async (req, res) => {
  const { texts, model } = req.body ?? {};
  if (!Array.isArray(texts) || !texts.length)
    return res.status(400).json({ error: "texts[] required" });

  const t0 = Date.now();
  const vecs = await embedBatch(texts, { model });
  res.json({
    embeddings: vecs.map((v) => Array.from(v)),
    dim: vecs[0]?.length,
    latencyMs: Date.now() - t0,
  });
});

/**
 * POST /rerank
 * Body: { query: string, candidates: [{id, text, ...}], topK?: number, diversity?: boolean }
 */
app.post("/rerank", async (req, res) => {
  const {
    query,
    candidates,
    topK = 10,
    diversity = false,
  } = req.body ?? {};

  if (!query) return res.status(400).json({ error: "query required" });
  if (!Array.isArray(candidates))
    return res.status(400).json({ error: "candidates[] required" });

  const t0 = Date.now();
  const results = diversity
    ? await rerankMMR(query, candidates, { topK })
    : await rerank(query, candidates, { topK });

  res.json({ results, latencyMs: Date.now() - t0 });
});

/**
 * GET /corpus/stats
 */
app.get("/corpus/stats", (_, res) => {
  const wordCounts = corpus.map((d) => d.text.split(/\s+/).length);
  const total = wordCounts.reduce((a, b) => a + b, 0);
  res.json({
    documents: corpus.length,
    totalWords: total,
    avgWords: corpus.length ? Math.round(total / corpus.length) : 0,
    minWords: Math.min(...wordCounts, 0),
    maxWords: Math.max(...wordCounts, 0),
  });
});

const PORT = process.env.PORT ?? 4001;
app.listen(PORT, () => {
  console.log(`\n🔍  pplx-retrieval-kernel`);
  console.log(`    http://localhost:${PORT}\n`);
});

export { app };
