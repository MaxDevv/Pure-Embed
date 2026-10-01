const DEFAULT_MODEL = "https://huggingface.co/blobbybob/potion-256d-compact-v3/resolve/main";
const DEFAULT_MAX_LENGTH = 512;

function cleanText(text) {
  let out = "";
  for (const char of text) {
    const cp = char.codePointAt(0);
    if (cp === 0 || cp === 0xfffd || (cp < 32 && cp !== 9 && cp !== 10 && cp !== 13) || (cp >= 127 && cp <= 159)) continue;
    if (/\s/u.test(char)) out += " ";
    else if (isChinese(cp)) out += ` ${char} `;
    else out += char;
  }
  return out;
}

function isChinese(cp) {
  return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);
}

function basicTokens(text) {
  // BERT normalization: clean controls, isolate CJK, lowercase, and strip accents.
  const normalized = cleanText(text).toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
  const words = [];
  let current = "";
  for (const char of normalized) {
    if (/\s/u.test(char)) {
      if (current) words.push(current);
      current = "";
    } else if (/\p{P}/u.test(char) || /[!-/:-@[-`{-~]/.test(char)) {
      if (current) words.push(current);
      current = "";
      words.push(char);
    } else current += char;
  }
  if (current) words.push(current);
  return words;
}

function tokenize(text, vocab, maxLength) {
  const ids = [];
  for (const word of basicTokens(text)) {
    const chars = Array.from(word);
    if (chars.length > 100) { ids.push(1); continue; }
    let start = 0;
    const pieces = [];
    let failed = false;
    while (start < chars.length) {
      let end = chars.length;
      let found;
      while (start < end) {
        const raw = chars.slice(start, end).join("");
        const candidate = start ? `##${raw}` : raw;
        if (vocab[candidate] !== undefined) { found = vocab[candidate]; break; }
        end--;
      }
      if (found === undefined) { failed = true; break; }
      pieces.push(found);
      start = end;
    }
    ids.push(...(failed ? [1] : pieces));
    if (ids.length >= maxLength) return ids.slice(0, maxLength);
  }
  return ids.slice(0, maxLength);
}

function parseSafetensors(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 8) throw new Error("Invalid safetensors file");
  const headerLength = Number(view.getBigUint64(0, true));
  if (headerLength > buffer.byteLength - 8) throw new Error("Invalid safetensors header length");
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const dataStart = 8 + headerLength;
  const tensor = header.embeddings;
  if (!tensor || !["I8", "F32"].includes(tensor.dtype) || tensor.shape.length !== 2) {
    throw new Error("Expected a 2D I8 or F32 'embeddings' tensor");
  }
  const [rows, dimensions] = tensor.shape;
  const begin = dataStart + tensor.data_offsets[0];
  const end = dataStart + tensor.data_offsets[1];
  const bytesPerValue = tensor.dtype === "I8" ? 1 : 4;
  if (end - begin !== rows * dimensions * bytesPerValue || end > buffer.byteLength) throw new Error("Invalid embedding tensor offsets");
  const weightTensor = header.weights;
  if (!weightTensor || weightTensor.dtype !== "F32" || weightTensor.shape[0] !== rows) throw new Error("Expected an F32 'weights' tensor matching the vocabulary");
  const weightsOffset = dataStart + weightTensor.data_offsets[0];
  if (weightTensor.data_offsets[1] - weightTensor.data_offsets[0] !== rows * 4) throw new Error("Invalid token weights tensor");
  return { header, rows, dimensions, offset: begin, dtype: tensor.dtype, weightsOffset };
}

async function get(url, fetchImpl) {
  const response = await fetchImpl(url);
  if (!response.ok) throw new Error(`Could not load ${url}: HTTP ${response.status}`);
  return response;
}

export class PotionEmbedder {
  constructor(vocab, weights, tokenWeights, rows, dimensions, normalize, maxLength) {
    this.vocab = vocab;
    this.weights = weights;
    this.tokenWeights = tokenWeights;
    this.rows = rows;
    this.dimensions = dimensions;
    this.normalize = normalize;
    this.maxLength = maxLength;
  }

  static async load({ model = DEFAULT_MODEL, maxLength = DEFAULT_MAX_LENGTH, fetch: fetchImpl = globalThis.fetch } = {}) {
    if (typeof fetchImpl !== "function") throw new Error("fetch is unavailable in this environment");
    const base = model.replace(/\/$/, "");
    const [tokenizerResponse, modelResponse, configResponse] = await Promise.all([
      get(`${base}/tokenizer.json`, fetchImpl), get(`${base}/model.safetensors`, fetchImpl), get(`${base}/config.json`, fetchImpl)
    ]);
    const [tokenizer, modelBuffer, config] = await Promise.all([
      tokenizerResponse.json(), modelResponse.arrayBuffer(), configResponse.json()
    ]);
    if (tokenizer.model?.type !== "WordPiece") throw new Error("This tokenizer is not WordPiece");
    const vocab = tokenizer.model.vocab;
    const { rows, dimensions, offset, dtype, weightsOffset } = parseSafetensors(modelBuffer);
    if (rows !== Object.keys(vocab).length) throw new Error(`Vocabulary has ${Object.keys(vocab).length} tokens but model has ${rows} rows`);
    const weights = dtype === "I8" ? new Int8Array(modelBuffer, offset, rows * dimensions) : new Float32Array(modelBuffer, offset, rows * dimensions);
    const tokenWeights = new Float32Array(modelBuffer, weightsOffset, rows);
    const normalize = config.normalize ?? true;
    return new PotionEmbedder(vocab, weights, tokenWeights, rows, dimensions, normalize, maxLength);
  }

  embed(text) {
    if (typeof text !== "string") throw new TypeError("embed(text) expects a string");
    const ids = tokenize(text, this.vocab, this.maxLength);
    const result = new Float32Array(this.dimensions);
    if (!ids.length) return result;
    for (const id of ids) {
      const start = id * this.dimensions;
      const scale = this.tokenWeights[id];
      for (let d = 0; d < this.dimensions; d++) result[d] += this.weights[start + d] * scale;
    }
    const inverseCount = 1 / ids.length;
    let normSquared = 0;
    for (let d = 0; d < result.length; d++) {
      result[d] *= inverseCount;
      normSquared += result[d] * result[d];
    }
    if (this.normalize) {
      const inverseNorm = 1 / Math.max(Math.sqrt(normSquared), 1e-12);
      for (let d = 0; d < result.length; d++) result[d] *= inverseNorm;
    }
    return result;
  }

  embedMany(texts) {
    if (!Array.isArray(texts)) throw new TypeError("embedMany(texts) expects an array of strings");
    return texts.map(text => this.embed(text));
  }
}

export async function load(options) {
  return PotionEmbedder.load(options);
}
