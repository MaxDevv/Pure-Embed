# Pure Embed

A lightweight library for **Fast, static text embeddings in the browser.** using pure JavaScript and a compact 8MB 256-dimensional model, with zero runtime dependencies. Fully client side.
*This was made with AI assistance, the whole librarys is just 160 lines tho*

Uses [Potion 256D Compact v3](https://huggingface.co/blobbybob/potion-256d-compact-v3), a Model2Vec static model. Inference is tokenization + table lookups + averaging: lightweight enough for phones and edge devices. The model files (~8.4 MB combined) are fetched once and cached by the browser/HTTP cache.

## Try it in HTML

Save this as `index.html` and serve it from a local or hosted HTTP server (ES modules and model downloads need HTTP; opening as `file://` is not supported):

```html
<!doctype html>
<html lang="en">
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pure Embed</title>
  <h1>Pure Embed — Find similar text</h1>
  <input id="query" value="A quick way to create text vectors in the browser" size="55">
  <button id="search">Compare</button>
  <pre id="result">Loading model…</pre>

  <script type="module">
    import { load } from "https://cdn.jsdelivr.net/gh/MaxDevv/Pure-Embed@master/src/index.js";

    const output = document.querySelector("#result");
    const model = await load(); // Downloads the model once; no API key or server required.
    const examples = [
      "Fast text embeddings run directly in your browser.",
      "A recipe for homemade tomato soup.",
      "Generate semantic vectors on a phone without sending text to a server."
    ];
    const vectors = model.embedMany(examples);
    const cosine = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

    document.querySelector("#search").addEventListener("click", () => {
      const query = model.embed(document.querySelector("#query").value);
      const matches = examples.map((text, i) => ({ text, score: cosine(query, vectors[i]) }))
        .sort((a, b) => b.score - a.score);
      output.textContent = matches.map(({ text, score }) => `${score.toFixed(3)}  ${text}`).join("\n");
    });
    output.textContent = "Model ready — click Compare.";
  </script>
</html>
```

> The CDN loads `src/index.js` from the public GitHub repository's `master` branch. To run a local checkout, replace the import with `"./src/index.js"` and serve the project directory (for example, `python -m http.server`).

## Install

```sh
npm install github:MaxDevv/Pure-Embed
```

```js
import { load } from "pure-embed";

const model = await load();
const vector = model.embed("Private text is embedded on this device.");
console.log(vector.length); // 256
```

That's the complete API. `embed()` returns a normalized `Float32Array`; `embedMany([...])` returns one for each input. No setup, API key, build step, server, or configuration required. The model is loaded lazily when `load()` is called.

## Options

```js
const model = await load({
  // Hugging Face model directory (defaults to Potion 256D Compact v3)
  model: "https://huggingface.co/blobbybob/potion-256d-compact-v3/resolve/main",
  maxLength: 512, // Maximum WordPiece tokens per text
  fetch: window.fetch // Optional custom fetch implementation
});
```

`load()` downloads the model's `tokenizer.json`, `model.safetensors`, and `config.json` in parallel. For offline/self-hosted use, mirror those three files and pass their directory URL as `model`. The model is MIT licensed; see its [model card](https://huggingface.co/blobbybob/potion-256d-compact-v3).

## Benchmark

Measured in browsers on October 1, 2026, using Potion 256D Compact v3:

| Device / browser | Input | Embeddings / second | Time / embedding |
| --- | --- | ---: | ---: |
| Base iPhone 14 / Safari | Short (6 words) | 172,685 | 5.8 µs |
| Base iPhone 14 / Safari | Medium (17 words) | 54,826 | 18.2 µs |
| Base iPhone 14 / Safari | Long (280 words) | 3,447 | 290.1 µs |
| Intel Core i7-11800H laptop / Chrome 154, Linux | Short (6 words) | 77,052 | 13.0 µs |
| Intel Core i7-11800H laptop / Chrome 154, Linux | Medium (17 words) | 27,795 | 36.0 µs |
| Intel Core i7-11800H laptop / Chrome 154, Linux | Long (280 words) | 1,939 | 515.7 µs |

Each case repeatedly embeds the same text on the browser's main thread, with 300 ms of warmup followed by five calibrated batches of approximately 250 ms. Results use the median batch time and include tokenization, table lookups, pooling, normalization, and output allocation. Input lengths are word counts. These measurements capture warmed-up throughput with repeated text; performance varies with text, browser, and device conditions.

Model download and initialization are excluded from throughput. Observed `load()` times were **2.08 s on the iPhone** and **1.52 s on the laptop**; cache state was uncontrolled. Both devices passed the 256-dimensional, unit-norm output check. Raw results are in [`benchmark/results/`](benchmark/results/).

To measure your own browser:

```sh
uv run python benchmark/serve.py
```

Open `http://localhost:8765`, enter your device name, and click **Run benchmark**. Keep the tab visible and disable Low Power Mode. Results are saved automatically to `benchmark/results/`. For a phone, expose the server with `ngrok http 8765` and open its HTTPS URL.

## Run and verify

```sh
npm test
```

The end-to-end test downloads the actual model and checks 256D normalized output, empty input, and semantic ranking. First run requires network access; subsequent requests use the Hugging Face/browser cache. The runtime itself has zero dependencies and is pure JavaScript.
