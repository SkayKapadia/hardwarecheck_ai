// Fetches HF config.json + safetensors index total_size for a list of repos.
// Params = total_size / bytes-per-param (from config torch_dtype).
const repos = [
  "Qwen/Qwen3-0.6B",
  "Qwen/Qwen3-1.7B",
  "Qwen/Qwen3-1.7B-Base",
  "Qwen/Qwen3-4B",
  "Qwen/Qwen3-4B-Base",
  "Qwen/Qwen3-4B-Instruct-2507",
  "Qwen/Qwen3-8B",
  "Qwen/Qwen3-14B",
  "Qwen/Qwen3-32B",
  "Qwen/Qwen2.5-0.5B",
  "Qwen/Qwen2.5-0.5B-Instruct",
  "Qwen/Qwen2.5-1.5B",
  "Qwen/Qwen2.5-1.5B-Instruct",
  "Qwen/Qwen2.5-3B-Instruct",
  "Qwen/Qwen2.5-7B-Instruct",
  "Qwen/Qwen2.5-Coder-7B-Instruct",
  "Qwen/Qwen2.5-14B-Instruct",
  "Qwen/Qwen2.5-Coder-14B-Instruct",
  "Qwen/Qwen2.5-32B-Instruct",
  "Qwen/Qwen2.5-Coder-32B-Instruct",
  "Qwen/Qwen3-30B-A3B",
  "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "Qwen/Qwen3-Coder-30B-A3B-Instruct",
  "mistralai/Mistral-7B-v0.1",
  "mistralai/Mistral-7B-Instruct-v0.2",
  "dphn/dolphin-2.9.1-yi-1.5-34b",
  "allenai/OLMo-2-0425-1B",
  "ibm-research/PowerMoE-3b",
  "deepseek-ai/DeepSeek-V3",
  "EleutherAI/pythia-70m-deduped",
  "EleutherAI/pythia-160m",
  "HuggingFaceTB/SmolLM2-135M",
  "openai-community/gpt2",
  "facebook/opt-125m",
  "distilbert/distilgpt2",
  // possibly-fictional future repos (expect 404s):
  "deepseek-ai/DeepSeek-V3.2",
  "deepseek-ai/DeepSeek-V4.1-Flash",
  "zai-org/GLM-5.2-FP8",
  "zai-org/GLM-4.7-Flash",
  "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
  "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
  "Qwen/Qwen3-Coder-Next-FP8",
  "MiniMaxAI/MiniMax-M2.7",
];

const BYTES = { bfloat16: 2, float16: 2, float32: 4, float: 4 };

async function getJSON(url) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) return null;
  return res.json();
}

const out = {};
for (const repo of repos) {
  try {
    const cfg = await getJSON(`https://huggingface.co/${repo}/raw/main/config.json`);
    if (!cfg) {
      out[repo] = { error: "no config.json (404?)" };
      console.error(`MISS ${repo}`);
      continue;
    }
    const idx = await getJSON(`https://huggingface.co/${repo}/raw/main/model.safetensors.index.json`);
    let params = null;
    if (idx?.metadata?.total_size) {
      const bpp = BYTES[cfg.torch_dtype] ?? 2;
      params = Math.round(idx.metadata.total_size / bpp);
    }
    out[repo] = {
      params,
      total_size: idx?.metadata?.total_size ?? null,
      torch_dtype: cfg.torch_dtype ?? null,
      layers: cfg.num_hidden_layers ?? null,
      hiddenSize: cfg.hidden_size ?? null,
      queryHeads: cfg.num_attention_heads ?? null,
      kvHeads: cfg.num_key_value_heads ?? null,
      vocabSize: cfg.vocab_size ?? null,
      maxContext: cfg.max_position_embeddings ?? null,
      intermediateSize: cfg.intermediate_size ?? null,
      moe: cfg.num_experts ?? cfg.n_routed_experts ?? cfg.num_local_experts ?? null,
    };
    console.error(`OK   ${repo} params=${params}`);
  } catch (e) {
    out[repo] = { error: String(e) };
    console.error(`ERR  ${repo}: ${e}`);
  }
}
console.log(JSON.stringify(out, null, 2));
