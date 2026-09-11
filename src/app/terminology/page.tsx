/* eslint-disable react/no-unescaped-entities */
import { BookOpen, Scale, Zap, HardDrive, Layers, PenTool, Search, Microscope, Network } from "lucide-react";

export default function TerminologyPage() {
  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-10 max-w-5xl mx-auto h-full">
      <h1 className="text-4xl font-bold text-foreground mb-4 flex items-center gap-4">
        <BookOpen className="w-10 h-10 text-primary" />
        Terminology & Methodology
      </h1>
      <p className="text-muted-foreground mb-12 text-lg leading-relaxed">
        A comprehensive guide to understanding AI models, hardware planning, and the differences between licensing types.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-12 pb-20">
        
        {/* Licensing Section */}
        <div className="space-y-6 md:col-span-2 bg-card p-8 rounded-xl border border-border">
          <h2 className="text-2xl font-bold text-foreground flex items-center gap-3">
            <Scale className="w-6 h-6 text-primary" />
            Licensing: Open Source vs. Open Weight
          </h2>
          <div className="space-y-6 text-foreground/80 leading-relaxed">
            <p>
              When discussing AI models, the terms "Open Source" and "Open Weight" are often used interchangeably, but they represent fundamentally different levels of access and licensing.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-background border border-border p-6 rounded-lg">
                <h3 className="font-bold text-lg text-primary mb-2">Open Source Models</h3>
                <p className="text-sm">
                  True open-source AI models provide full access to not just the final model weights, but the <strong>entire training pipeline</strong>. This includes the massive datasets used to train the model, the data filtering code, the training scripts, and the model architecture. 
                  <br/><br/>
                  <em>Examples: OLMo (AI2), Pythia (EleutherAI).</em>
                </p>
              </div>
              <div className="bg-background border border-border p-6 rounded-lg">
                <h3 className="font-bold text-secondary mb-2 text-lg">Open Weight Models</h3>
                <p className="text-sm">
                  Most modern commercial "open" models are actually open-weight. The creators release the final, pre-trained neural network parameters (the "weights") so anyone can download and run the model locally. However, they keep the training data and training methodology a closely guarded secret.
                  <br/><br/>
                  <em>Examples: Llama 3 (Meta), Mistral, Qwen.</em>
                </p>
              </div>
            </div>
            
            <h3 className="font-bold text-foreground mt-4">Implementation Differences</h3>
            <p className="text-sm">
              <strong>Implementing an Open Weight Model:</strong> You download the pre-trained weights (e.g., a `.safetensors` or `.gguf` file) and run it using an inference engine like vLLM or llama.cpp. You can fine-tune it (using LoRA) on your own small dataset, but you cannot fundamentally change its base knowledge because you don't have the original training data.
            </p>
            <p className="text-sm">
              <strong>Implementing an Open Source Model:</strong> You have the ability to replicate the exact model from scratch, alter the training data to remove biases, or continue pre-training exactly where the original authors left off. It offers total scientific reproducibility.
            </p>
          </div>
        </div>

        {/* Technical Terminology */}
        <div className="space-y-4 bg-card p-8 rounded-xl border border-border">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-3 border-b border-border pb-2">
            <Layers className="w-5 h-5 text-primary" />
            Model Architecture
          </h2>
          <div className="space-y-4">
            <div>
              <h3 className="font-bold text-foreground">Parameters (e.g., 8B, 70B)</h3>
              <p className="text-sm text-muted-foreground mt-1">The total number of neural connections (weights) in the model. An "8B" model has 8 Billion parameters. More parameters generally equal higher intelligence but require exponentially more VRAM to run.</p>
            </div>
            <div>
              <h3 className="font-bold text-foreground">Quantization</h3>
              <p className="text-sm text-muted-foreground mt-1">A technique to compress model weights. Standard weights are 16-bit (FP16). Quantization compresses them to 8-bit, 4-bit, or even 2-bit formats. This drastically reduces VRAM requirements and increases generation speed, at the cost of a slight reduction in reasoning ability.</p>
            </div>
            <div>
              <h3 className="font-bold text-foreground">Dense vs MoE</h3>
              <p className="text-sm text-muted-foreground mt-1"><strong>Dense</strong> models activate every single parameter for every word generated. <strong>Mixture of Experts (MoE)</strong> models split their parameters into "expert" groups and only activate a fraction of them per word, resulting in massive models that run surprisingly fast.</p>
            </div>
          </div>
        </div>

        <div className="space-y-4 bg-card p-8 rounded-xl border border-border">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-3 border-b border-border pb-2">
            <HardDrive className="w-5 h-5 text-primary" />
            Memory & Inference
          </h2>
          <div className="space-y-4">
            <div>
              <h3 className="font-bold text-foreground">KV Cache</h3>
              <p className="text-sm text-muted-foreground mt-1">The "Key-Value Cache". As an AI reads your prompt and generates text, it saves the mathematical representations of previous words in VRAM so it doesn't have to recalculate them. A large context length requires a massive KV Cache.</p>
            </div>
            <div>
              <h3 className="font-bold text-foreground">Context Length</h3>
              <p className="text-sm text-muted-foreground mt-1">The maximum number of "tokens" (words or sub-words) the AI can remember in a single interaction. 8,192 tokens is roughly equivalent to 6,000 words.</p>
            </div>
            <div>
              <h3 className="font-bold text-foreground">Batch Size</h3>
              <p className="text-sm text-muted-foreground mt-1">The number of independent conversations the model is processing simultaneously. A batch size of 1 means a single user is talking to the AI. A batch size of 32 means a server is handling 32 simultaneous users.</p>
            </div>
          </div>
        </div>

        {/* Advanced Data Scientist Concepts */}
        <div className="space-y-6 md:col-span-2 bg-card p-8 rounded-xl border border-border mt-4">
          <h2 className="text-2xl font-bold text-foreground flex items-center gap-3">
            <Microscope className="w-6 h-6 text-primary" />
            Advanced AI Science & Distributed Systems
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-foreground/80 leading-relaxed mt-4">
            
            <div className="space-y-4">
              <div>
                <h3 className="font-bold text-foreground">Datatypes: BF16 vs FP16 vs FP32</h3>
                <p className="text-sm text-muted-foreground mt-1">AI scientists care deeply about precision. <strong>FP32</strong> (32-bit Float) offers extreme precision but wastes memory. <strong>FP16</strong> halves memory but suffers from overflow/underflow during training. <strong>BF16 (Brain Float 16)</strong> sacrifices some decimal precision for a larger exponent range, making it the absolute gold-standard for stable AI training.</p>
              </div>
              <div>
                <h3 className="font-bold text-foreground">Optimizer States (AdamW)</h3>
                <p className="text-sm text-muted-foreground mt-1">When training a model, the optimizer (like AdamW) keeps track of momentum and variance for every single parameter. This means the optimizer actually consumes <strong>2x to 3x more VRAM</strong> than the model weights themselves! This is why a model that fits in 16GB for inference might require 80GB for training.</p>
              </div>
              <div>
                <h3 className="font-bold text-foreground">Gradient Checkpointing</h3>
                <p className="text-sm text-muted-foreground mt-1">A critical memory-saving technique used during training. Instead of storing all the intermediate neural activations during the forward pass (which requires massive VRAM), gradient checkpointing throws them away and simply <em>recalculates</em> them during the backward pass. It trades 20% slower training time for up to 60% less VRAM usage.</p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="font-bold text-foreground">GQA vs MQA vs MHA</h3>
                <p className="text-sm text-muted-foreground mt-1">Attention mechanisms define how a model thinks. <strong>MHA (Multi-Head Attention)</strong> gives every query its own Key and Value, which creates a massive, memory-hogging KV Cache. <strong>MQA (Multi-Query Attention)</strong> forces all queries to share a single Key/Value, saving huge amounts of memory. <strong>GQA (Grouped-Query Attention)</strong> is the modern compromise (used in Llama 3), grouping queries to balance intelligence and memory.</p>
              </div>
              <div>
                <h3 className="font-bold text-foreground flex items-center gap-2"><Network className="w-4 h-4" /> Tensor Parallelism (TP)</h3>
                <p className="text-sm text-muted-foreground mt-1">Used to split a single massive model across multiple GPUs. TP physically slices individual matrix multiplications (tensors) in half. GPU 1 calculates the left half of the matrix, GPU 2 calculates the right half, and they sync over NVLink. It requires extreme bandwidth.</p>
              </div>
              <div>
                <h3 className="font-bold text-foreground flex items-center gap-2"><Network className="w-4 h-4" /> FSDP / Pipeline Parallelism (PP)</h3>
                <p className="text-sm text-muted-foreground mt-1">Unlike TP, <strong>Pipeline Parallelism</strong> puts layer 1-10 on GPU 1, and layer 11-20 on GPU 2, passing the data sequentially. <strong>FSDP (Fully Sharded Data Parallel)</strong> splits the model weights, optimizer states, and gradients evenly across hundreds of GPUs, gathering them only exactly when needed for a calculation.</p>
              </div>
            </div>

          </div>
        </div>

        {/* Workflows */}
        <div className="space-y-4 bg-card p-8 rounded-xl border border-border md:col-span-2">
          <h2 className="text-xl font-bold text-foreground flex items-center gap-3 border-b border-border pb-2">
            <PenTool className="w-5 h-5 text-primary" />
            Implementation Workflows
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
            <div>
              <h3 className="font-bold text-foreground flex items-center gap-2">
                <Zap className="w-4 h-4 text-emerald-500" /> Inference
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                Simply running the model to generate text. The model's weights are "frozen" and cannot be changed. This requires the least amount of VRAM because you only need to store the Weights and the KV Cache.
              </p>
            </div>
            <div>
              <h3 className="font-bold text-foreground flex items-center gap-2">
                <PenTool className="w-4 h-4 text-purple-500" /> Fine-Tuning (LoRA)
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                Low-Rank Adaptation. Instead of retraining the entire massive model, you inject small "adapter" layers into the model and train only those. This allows you to teach the model a specific tone or format on consumer hardware without catastrophic memory usage.
              </p>
            </div>
            <div>
              <h3 className="font-bold text-foreground flex items-center gap-2">
                <Search className="w-4 h-4 text-amber-500" /> RAG
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                Retrieval-Augmented Generation. Instead of fine-tuning a model to "memorize" a textbook, you store the textbook in a vector database. When a user asks a question, you search the database for the relevant paragraph and paste it into the model's prompt. This is infinitely cheaper and more accurate than fine-tuning for knowledge retention.
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
