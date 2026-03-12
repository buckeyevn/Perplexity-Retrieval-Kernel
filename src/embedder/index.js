/**
 * embedder/index.js
 *
 * Text embedding client for pplx-retrieval-kernel.
 *
 * Architecture:
 *   - Primary: internal pplx-embed-v3 endpoint (E5-mistral-7b fine-tuned)
 *   - Fallback: OpenAI text-embedding-3-large
 *   - Batching: up to BATCH_SIZE texts per request, parallelized
 *   - LRU cache keyed on SHA-256(text + model)
 *
 * Embedding dimensions:
 *   pplx-embed-v3        : 4096 → projected to 1024
 *   text-embedding-3-large: 3072 → projected to 1024
 */

import { createHash } from "crypto";
import { LRUCache } from "lru-cache";

const BATCH_SIZE = parseInt(process.env.EMBED_BATCH_SIZE ?? "32", 10);
const EMBED_DIM = 1024;
const EMBED_ENDPOINT =
  process.env.PPLX_EMBED_ENDPOINT ?? "http://embed-svc.pplx.internal:8080";
const CACHE_MAX = parseInt(process.env.EMBED_CACHE_MAX ?? "50000", 10);

/** @type {LRUCache<string, Float32Array>} */
const cache = new LRUCache({ max: CACHE_MAX });

/**
 * Hash a text+model pair for cache keying.
 * @param {string} text
 * @param {string} model
 * @returns {string}
 */
function cacheKey(text, model) {
  return createHash("sha256").update(`${model}::${text}`).digest("hex");
}

/**
 * Simulate the internal embedding call.
 * In production this hits pplx-embed-v3 over gRPC.
 * Here we produce deterministic pseudo-embeddings for testing.
 *
 * @param {string[]} texts
 * @param {string}   model
 * @returns {Promise<Float32Array[]>}
 */
async function callEmbedService(texts, model = "pplx-embed-v3") {
  // Simulate latency of the real service
  await new Promise((r) => setTimeout(r, 2 + texts.length * 0.5));

  return texts.map((text) => {
    // Deterministic pseudo-embedding based on text hash
    const hash = createHash("sha256").update(text).digest();
    const vec = new Float32Array(EMBED_DIM);
    for (let i = 0; i < EMBED_DIM; i++) {
      // Spread hash bytes across dimensions with some variation
      vec[i] = ((hash[i % 32] / 255) - 0.5) * 2 + Math.sin(i * 0.1) * 0.01;
    }
    // L2 normalize
    let norm = 0;
    for (let i = 0; i < EMBED_DIM; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < EMBED_DIM; i++) vec[i] /= norm;
    return vec;
  });
}

/**
 * Embed a single text string.
 * @param {string} text
 * @param {Object} [opts]
 * @param {string} [opts.model]
 * @returns {Promise<Float32Array>}
 */
export async function embed(text, { model = "pplx-embed-v3" } = {}) {
  const key = cacheKey(text, model);
  if (cache.has(key)) return cache.get(key);

  const [vec] = await callEmbedService([text], model);
  cache.set(key, vec);
  return vec;
}

/**
 * Embed multiple texts with batching and cache deduplication.
 * @param {string[]} texts
 * @param {Object}   [opts]
 * @param {string}   [opts.model]
 * @returns {Promise<Float32Array[]>}
 */
export async function embedBatch(texts, { model = "pplx-embed-v3" } = {}) {
  const results = new Array(texts.length);
  const uncachedIndices = [];
  const uncachedTexts = [];

  // Check cache first
  for (let i = 0; i < texts.length; i++) {
    const key = cacheKey(texts[i], model);
    if (cache.has(key)) {
      results[i] = cache.get(key);
    } else {
      uncachedIndices.push(i);
      uncachedTexts.push(texts[i]);
    }
  }

  if (uncachedTexts.length === 0) return results;

  // Chunk uncached texts into batches
  const batches = [];
  for (let i = 0; i < uncachedTexts.length; i += BATCH_SIZE) {
    batches.push(uncachedTexts.slice(i, i + BATCH_SIZE));
  }

  const batchResults = (
    await Promise.all(batches.map((b) => callEmbedService(b, model)))
  ).flat();

  // Store results and populate cache
  for (let i = 0; i < uncachedIndices.length; i++) {
    const originalIdx = uncachedIndices[i];
    const vec = batchResults[i];
    results[originalIdx] = vec;
    cache.set(cacheKey(texts[originalIdx], model), vec);
  }

  return results;
}

/**
 * Cosine similarity between two unit vectors.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {number}
 */
export function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // vectors are pre-normalized
}

export function getCacheStats() {
  return { size: cache.size, max: CACHE_MAX, hitRate: cache.calculatedSize };
}

export { EMBED_DIM };
