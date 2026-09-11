import fs from 'fs';

const data = JSON.parse(fs.readFileSync('src/data/models.json', 'utf8'));

// 1. Filter out dummy models
let cleaned = data.filter(m => {
  const n = m.name.toLowerCase();
  return !n.includes("tiny") && !n.includes("random") && !n.includes("test");
});

// 2. Fix MoE and Parameter Counts
for (let m of cleaned) {
  // Qwen3 30B A3B
  if (m.name.includes("30B A3B")) {
    m.architecture = "moe";
    m.activeParams = 3000000000;
  }
  // DeepSeek V3 / V4
  if (m.id.includes("DeepSeek-V3") || m.id.includes("DeepSeek-V4")) {
    m.architecture = "moe";
    m.activeParams = 37000000000;
  }
  // Nemotron 120B A12B
  if (m.name.includes("120B A12B")) {
    m.architecture = "moe";
    m.activeParams = 12000000000;
  }
  // Qwen 72B (Listed as 120B)
  if (m.name === "Qwen 72B" && m.params > 100000000000) {
    m.params = 72000000000;
  }
  // GPT-2 Large
  if (m.id === "openai-community/gpt2-large") {
    m.params = 774000000;
  }
  // Llama 3.1 8B
  if (m.id === "meta-llama/Meta-Llama-3.1-8B-Instruct") {
    m.params = 8000000000;
  }
}

fs.writeFileSync('src/data/models.json', JSON.stringify(cleaned, null, 2));
console.log(`Cleaned dataset from ${data.length} to ${cleaned.length} models.`);
