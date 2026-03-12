/**
 * rerank/index.js
 *
 * Cross-encoder reranker — second-stage scoring of retrieved candidates.
 *
 * Architecture:
 *   Stage 1 (ranker): BM25 + dense ANN → top-50 candidates  (cheap, fast)
 *   Stage 2 (rerank): cross-encoder attention on [query, passage] → top-k  (accurate, slower)
 *
 * The cross-encoder sees the full (query, document) pair jointly,
 * unlike bi-encoders which score them independently.
 * This catches relevance signals missed by embedding similarity.
 *
 * In production: pplx-rerank-v2 (fine-tuned on Perplexity query logs)
 * Here: simulated with a learned score approximation for testing.
 */

import { embed, cosineSim } from "../embedder/index.js";

const RERANK_ENDPOINT =
  process.env.PPLX_RERANK_ENDPOINT ?? "http://rerank-svc.pplx.internal:8081";

/**
 * Simulated cross-encoder score.
 * Real implementation calls pplx-rerank-v2 via gRPC.
 *
 * Scoring factors:
 *   - Term overlap (lexical precision)
 *   - Embedding similarity
 *   - Document length penalty (prefer focused passages)
 *   - Query term density
 */
async function crossEncoderScore(query, passage) {
  await new Promise((r) => setTimeout(r, 0.5)); // simulate inference

  const qTerms = new Set(
    query.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean)
  );
  const pWords = passage
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(/\s+/)
    .filter(Boolean);

  // Term overlap
  const overlap = pWords.filter((w) => qTerms.has(w)).length;
  const termScore = Math.min(overlap / Math.max(qTerms.size, 1), 1.0);

  // Query term density
  const density = overlap / Math.max(pWords.length, 1);

  // Length penalty: prefer 50-300 word passages
  const wc = pWords.length;
  const lenPenalty = wc < 10 ? 0.5 : wc > 500 ? 0.8 : 1.0;

  // Embedding similarity (async)
  const qVec = await embed(query);
  const pVec = await embed(passage.slice(0, 512)); // truncate for speed
  const embSim = (cosineSim(qVec, pVec) + 1) / 2; // [0,1]

  // Weighted combination (mirrors cross-encoder logit decomposition)
  return (0.35 * termScore + 0.25 * density + 0.40 * embSim) * lenPenalty;
}

/**
 * Rerank a list of candidates for a query.
 *
 * @param {string} query
 * @param {Array<{id:string, text:string, [key:string]:any}>} candidates
 * @param {Object} [opts]
 * @param {number} [opts.topK]           - return top-k after reranking
 * @param {number} [opts.concurrency]    - parallel scoring workers
 * @returns {Promise<Array<{id:string, text:string, rerankScore:number, originalRank:number}>>}
 */
export async function rerank(query, candidates, { topK = 10, concurrency = 8 } = {}) {
  if (candidates.length === 0) return [];

  // Score in parallel with concurrency limit
  const limit = (await import("p-limit")).default(concurrency);
  const scored = await Promise.all(
    candidates.map((c, i) =>
      limit(async () => {
        const score = await crossEncoderScore(query, c.text);
        return { ...c, rerankScore: score, originalRank: i + 1 };
      })
    )
  );

  return scored.sort((a, b) => b.rerankScore - a.rerankScore).slice(0, topK);
}

/**
 * Rerank with diversity: penalize candidates similar to already-selected ones.
 * Implements Maximal Marginal Relevance (MMR).
 *
 * @param {string}  query
 * @param {Array}   candidates
 * @param {Object}  [opts]
 * @param {number}  [opts.topK]
 * @param {number}  [opts.lambda]  - 0=max diversity, 1=max relevance
 */
export async function rerankMMR(
  query,
  candidates,
  { topK = 10, lambda = 0.7 } = {}
) {
  if (candidates.length === 0) return [];

  // Embed all candidates
  const vecs = await (
    await import("../embedder/index.js")
  ).embedBatch(candidates.map((c) => c.text.slice(0, 512)));

  const qVec = await embed(query);
  const relevance = vecs.map((v) => (cosineSim(qVec, v) + 1) / 2);

  const selected = [];
  const remaining = candidates.map((c, i) => ({ ...c, idx: i }));

  while (selected.length < topK && remaining.length > 0) {
    let bestScore = -Infinity;
    let bestIdx = 0;

    for (let i = 0; i < remaining.length; i++) {
      const rel = relevance[remaining[i].idx];

      // Max similarity to already-selected
      let maxSim = 0;
      for (const sel of selected) {
        const sim = (cosineSim(vecs[remaining[i].idx], vecs[sel.idx]) + 1) / 2;
        if (sim > maxSim) maxSim = sim;
      }

      const mmrScore = lambda * rel - (1 - lambda) * maxSim;
      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIdx = i;
      }
    }

    selected.push({ ...remaining[bestIdx], mmrScore: bestScore });
    remaining.splice(bestIdx, 1);
  }

  return selected.map((s, rank) => ({ ...s, rerankRank: rank + 1 }));
}
