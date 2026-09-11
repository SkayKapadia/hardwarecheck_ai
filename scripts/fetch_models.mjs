import fs from 'fs/promises';

const HF_API_URL = "https://huggingface.co/api/models";
const TOP_N = 100;

async function fetchTopModels() {
  console.log("Fetching top models from Hugging Face...");
  // Fetch top text-generation models sorted by downloads
  const res = await fetch(`${HF_API_URL}?filter=text-generation&sort=downloads&direction=-1&limit=${TOP_N}`);
  if (!res.ok) throw new Error("Failed to fetch models list");
  const models = await res.json();
  return models.map(m => m.id);
}

async function fetchModelConfig(modelId) {
  const res = await fetch(`https://huggingface.co/${modelId}/raw/main/config.json`);
  if (!res.ok) return null;
  try {
    return await res.json();
  } catch (e) {
    return null;
  }
}

function parseModelSpec(modelId, config) {
  const archType = config.architectures?.[0];
  if (!archType) return null;

  // Generic extraction logic, handling variations across architectures
  const layers = config.num_hidden_layers || config.n_layer || config.n_layers;
  const hiddenSize = config.hidden_size || config.n_embd || config.d_model;
  const queryHeads = config.num_attention_heads || config.n_head || config.n_heads;
  const kvHeads = config.num_key_value_heads || config.n_kv_head || config.multi_query_group_num || queryHeads;
  const vocabSize = config.vocab_size;
  const maxContext = config.max_position_embeddings || config.n_positions || config.max_seq_len;
  
  // Exclude models where we can't figure out basic specs
  if (!layers || !hiddenSize || !queryHeads || !vocabSize || !maxContext) return null;

  let architecture = "dense";
  let activeParams = undefined;

  // Basic check for MoE
  if (config.num_local_experts || config.expert_model || archType.includes("MoE") || archType.includes("Mixtral")) {
    architecture = "moe";
    // rough active params estimate if not provided: usually 1/3 to 1/4 of total params for 8 expert models
  }

  // Parameter calculation is complex without safetensors metadata, 
  // but we can estimate roughly from hidden size and layers
  // Parameters = Embedding + (Layers * (Attention + MLP)) + LM Head
  // Embedding = vocab_size * hidden_size
  // Attention = 4 * hidden_size^2 (if dense, less for GQA)
  // MLP = usually 8/3 * hidden_size^2
  
  // A rough heuristic just for demonstration if we don't fetch safetensors index:
  const isGQA = kvHeads < queryHeads;
  const headDim = hiddenSize / queryHeads;
  const attnParams = (hiddenSize * hiddenSize) + (3 * hiddenSize * (isGQA ? (kvHeads * headDim) : hiddenSize));
  
  let intermediateSize = config.intermediate_size || config.n_inner || (hiddenSize * 4);
  const mlpParams = 3 * hiddenSize * intermediateSize;
  
  let totalParams = (vocabSize * hiddenSize) + (layers * (attnParams + mlpParams)) + (vocabSize * hiddenSize);
  
  if (architecture === "moe" && config.num_local_experts) {
     totalParams = (vocabSize * hiddenSize) + (layers * (attnParams + (mlpParams * config.num_local_experts))) + (vocabSize * hiddenSize);
     activeParams = (vocabSize * hiddenSize) + (layers * (attnParams + (mlpParams * (config.num_experts_per_tok || 2)))) + (vocabSize * hiddenSize);
  }

  const name = modelId.split('/').pop().replace(/-/g, ' ');

  return {
    id: modelId,
    name,
    params: Math.round(totalParams),
    layers,
    hiddenSize,
    queryHeads,
    kvHeads,
    vocabSize,
    maxContext,
    architecture,
    activeParams: activeParams ? Math.round(activeParams) : undefined
  };
}

async function run() {
  const modelIds = await fetchTopModels();
  const validModels = [];
  
  console.log(`Processing ${modelIds.length} models...`);
  
  // Batch processing to avoid rate limits
  for (let i = 0; i < modelIds.length; i += 5) {
    const batch = modelIds.slice(i, i + 5);
    const promises = batch.map(async (id) => {
      try {
         const config = await fetchModelConfig(id);
         if (config) {
            const spec = parseModelSpec(id, config);
            if (spec) return spec;
         }
      } catch (e) {
        // ignore fetch errors
      }
      return null;
    });
    
    const results = await Promise.all(promises);
    validModels.push(...results.filter(Boolean));
    process.stdout.write(`\rProcessed ${Math.min(i + 5, modelIds.length)} / ${modelIds.length}`);
    await new Promise(r => setTimeout(r, 500)); // Sleep 500ms
  }
  
  console.log(`\nFound ${validModels.length} models with parsable config.json`);
  
  // Combine with our existing curated models to ensure we don't lose the high quality ones
  const existingPath = './src/data/models.json';
  const existingRaw = await fs.readFile(existingPath, 'utf8');
  const existingModels = JSON.parse(existingRaw);
  
  const finalModelsMap = new Map();
  // Add existing first (prioritize their clean data)
  for (const m of existingModels) {
    finalModelsMap.set(m.id, m);
  }
  // Add new fetched models
  for (const m of validModels) {
    if (!finalModelsMap.has(m.id)) {
      finalModelsMap.set(m.id, m);
    }
  }
  
  const finalArray = Array.from(finalModelsMap.values());
  await fs.writeFile(existingPath, JSON.stringify(finalArray, null, 2));
  console.log(`Successfully updated ${existingPath} with ${finalArray.length} total models.`);
}

run().catch(console.error);
