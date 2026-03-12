/**
 * ranker/index.js
 *
 * Hybrid retrieval: BM25 lexical score + neural embedding score.
 *
 * Score = α * bm25(q, d) + (1-α) * cosine(embed(q), embed(d))
 *
 * α is tuned per query type:
 *   - factual/entity queries: α=0.3  (lean neural)
 *   - keyword/technical:      α=0.6  (lean BM25)
 *   - conversational:         α=0.2  (lean neural)
 */

import { embedBatch, cosineSim } from "../embedder/index.js";

const DEFAULT_K1 = 1.5;
const DEFAULT_B = 0.75;

/**
 * Minimal in-memory BM25 index.
 */
export class BM25Index {
  /**
   * @param {Array<{id:string, text:string}>} documents
   * @param {Object} [opts]
   * @param {number} [opts.k1]
   * @param {number} [opts.b]
   */
  constructor(documents, { k1 = DEFAULT_K1, b = DEFAULT_B } = {}) {
    this.k1 = k1;
    this.b = b;
    this.docs = documents;
    this._build();
  }

  _tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);
  }

  _build() {
    const N = this.docs.length;
    this.termFreqs = [];
    this.docLengths = [];
    const df = new Map();

    for (const doc of this.docs) {
      const tokens = this._tokenize(doc.text);
      this.docLengths.push(tokens.length);
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      this.termFreqs.push(tf);
      for (const t of tf.keys()) df.set(t, (df.get(t) ?? 0) + 1);
    }

    this.avgDl = this.docLengths.reduce((a, b) => a + b, 0) / N || 1;

    // IDF map
    this.idf = new Map();
    for (const [term, freq] of df) {
      this.idf.set(term, Math.log((N - freq + 0.5) / (freq + 0.5) + 1));
    }
  }

  /**
   * Score all documents against a query string.
   * @param {string} query
   * @returns {Array<{id:string, score:number}>} sorted descending
   */
  score(query) {
    const qTokens = this._tokenize(query);
    const scores = this.docs.map((doc, i) => {
      let s = 0;
      const tf = this.termFreqs[i];
      const dl = this.docLengths[i];

      for (const term of qTokens) {
        const freq = tf.get(term) ?? 0;
        if (freq === 0) continue;
        const idf = this.idf.get(term) ?? 0;
        const num = freq * (this.k1 + 1);
        const den = freq + this.k1 * (1 - this.b + this.b * (dl / this.avgDl));
        s += idf * (num / den);
      }

      return { id: doc.id, score: s };
    });

    return scores.sort((a, b) => b.score - a.score);
  }
}

/**
 * Hybrid retriever combining BM25 and dense embeddings.
 */
export class HybridRetriever {
  /**
   * @param {Array<{id:string, text:string}>} documents
   * @param {Object} [opts]
   * @param {number} [opts.alpha]  - BM25 weight (0=full neural, 1=full BM25)
   * @param {string} [opts.model]  - embedding model
   */
  constructor(documents, { alpha = 0.35, model = "pplx-embed-v3" } = {}) {
    this.documents = documents;
    this.alpha = alpha;
    this.model = model;
    this.bm25 = new BM25Index(documents);
    this._docVecs = null; // lazy-initialized
  }

  async _ensureVecs() {
    if (this._docVecs) return;
    this._docVecs = await embedBatch(
      this.documents.map((d) => d.text),
      { model: this.model }
    );
  }

  /**
   * Retrieve top-k documents for a query.
   * @param {string} query
   * @param {number} [k]
   * @param {number} [alpha]  - override instance alpha
   * @returns {Promise<Array<{id:string, text:string, score:number, bm25:number, neural:number}>>}
   */
  async retrieve(query, k = 10, alpha = this.alpha) {
    await this._ensureVecs();

    // BM25 scores (already sorted)
    const bm25Scores = this.bm25.score(query);
    const bm25Map = new Map(bm25Scores.map((s) => [s.id, s.score]));

    // Normalize BM25 scores to [0,1]
    const bm25Max = bm25Scores[0]?.score ?? 1;
    const bm25Norm = bm25Max > 0 ? bm25Max : 1;

    // Neural scores
    const queryVec = await (
      await import("../embedder/index.js")
    ).embed(query, { model: this.model });
    const neuralMap = new Map(
      this.documents.map((d, i) => [d.id, cosineSim(queryVec, this._docVecs[i])])
    );

    // Hybrid score
    const scored = this.documents.map((doc) => {
      const bm25 = (bm25Map.get(doc.id) ?? 0) / bm25Norm;
      const neural = (neuralMap.get(doc.id) ?? 0 + 1) / 2; // [-1,1] → [0,1]
      const score = alpha * bm25 + (1 - alpha) * neural;
      return { ...doc, score, bm25, neural };
    });

    return scored.sort((a, b) => b.score - a.score).slice(0, k);
  }

  /** Update alpha based on query classification */
  setAlphaForQueryType(type) {
    const alphaMap = {
      factual: 0.3,
      keyword: 0.6,
      conversational: 0.2,
      technical: 0.55,
    };
    this.alpha = alphaMap[type] ?? 0.35;
  }
}
