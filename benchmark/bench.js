import { performance } from "node:perf_hooks";
import { load } from "../src/index.js";

const rounds = Number(process.env.ROUNDS ?? 5);
const cases = [
  { name: "short (~6 tokens)", text: "Fast local text search works well.", iterations: 20_000 },
  { name: "medium (~20 tokens)", text: "A small static embedding model turns this sentence into a useful vector directly on the user's device.", iterations: 10_000 },
  { name: "long (~90 tokens)", text: Array(10).fill("A static embedding model uses a compact vocabulary lookup table to quickly represent text as a normalized semantic vector without a neural network or a remote inference service.").join(" "), iterations: 2_000 }
];

console.log("Loading Potion 256D Compact v3…");
const model = await load();
console.log(`Loaded ${model.rows.toLocaleString()} tokens × ${model.dimensions} dimensions.\n`);

for (const { name, text, iterations } of cases) {
  for (let i = 0; i < 2_000; i++) model.embed(text); // warm up JIT
  const times = [];
  for (let run = 0; run < rounds; run++) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) model.embed(text);
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  const medianMs = times[Math.floor(times.length / 2)];
  console.log(`${name.padEnd(23)} ${Math.round(iterations / (medianMs / 1000)).toLocaleString()} embeds/s  (${(medianMs * 1000 / iterations).toFixed(1)} µs/embed; median of ${rounds})`);
}
