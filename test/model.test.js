import test from "node:test";
import assert from "node:assert/strict";
import { load } from "../src/index.js";

test("real Potion model returns normalized 256D semantic embeddings", async () => {
  const model = await load();
  const [relatedA, relatedB, unrelated] = model.embedMany([
    "A fast browser embedding model.",
    "A speedy in-browser model for text vectors.",
    "The weather is sunny today."
  ]);
  assert.equal(model.dimensions, 256);
  assert.equal(model.rows, 29525);
  for (const vector of [relatedA, relatedB, unrelated]) {
    assert.equal(vector.length, 256);
    assert.ok(Math.abs(Math.hypot(...vector) - 1) < 1e-5);
  }
  const cosine = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
  assert.ok(cosine(relatedA, relatedB) > cosine(relatedA, unrelated));
  assert.ok(Math.abs(relatedA[0] - (-0.02542093)) < 1e-6);
  assert.deepEqual([...model.embed("")], Array(256).fill(0));
});
