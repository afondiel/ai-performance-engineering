> *"Premature optimization is the root of all evil - but we should not pass up opportunities in the critical 3%."* - Donald Knuth

[![](https://img.shields.io/badge/Contribute-Welcome-green)](./CONTRIBUTING.md) ![GitHub](https://img.shields.io/github/license/afondiel/ai-performance-engineering)

# AI Performance Engineering Cheatsheet

> **From Cloud to Edge** — A holistic reference covering hardware fundamentals, GPU/accelerator metrics, the roofline model, AI/ML training & inference KPIs, LLM serving, distributed systems, quantization, cloud-native & edge deployment, networking, compiler optimizations, and practical tooling.

---

**Table of Contents**

- [1. Core Execution Metrics (CPU)](#1-core-execution-metrics-cpu)
- [2. GPU & Accelerator Metrics](#2-gpu--accelerator-metrics)
- [3. Time, Throughput & Latency](#3-time-throughput--latency)
- [4. Roofline Model](#4-roofline-model)
- [5. Parallel Performance](#5-parallel-performance)
- [6. Memory & I/O Metrics](#6-memory--io-metrics)
- [7. System-Level & Queueing Metrics](#7-system-level--queueing-metrics)
- [8. AI/ML Training Metrics](#8-aiml-training-metrics)
- [9. AI/ML Inference & Serving Metrics](#9-aiml-inference--serving-metrics)
- [10. LLM-Specific Performance Metrics](#10-llm-specific-performance-metrics)
- [11. Distributed Training & Multi-Device Scaling](#11-distributed-training--multi-device-scaling)
- [12. Quantization & Numerical Precision](#12-quantization--numerical-precision)
- [13. Network & Interconnect Performance](#13-network--interconnect-performance)
- [14. Cloud Performance Engineering](#14-cloud-performance-engineering)
- [15. Edge & Embedded AI Performance](#15-edge--embedded-ai-performance)
- [16. Compiler & Runtime Optimizations](#16-compiler--runtime-optimizations)
- [17. Energy, Cost & Sustainability](#17-energy-cost--sustainability)
- [18. Tooling & Profiling Reference](#18-tooling--profiling-reference)
- [19. Quick Practical Summary](#19-quick-practical-summary)
- [20. References](#20-references)

---

## 1. Core Execution Metrics (CPU)

### 1.1 Clock Frequency

- **Symbol:** $f_{\text{clock}}$
- **Unit:** Hz (cycles per second), often GHz
- **Meaning:** How many clock cycles the processor completes per second.

Example:
- 3.2 GHz => $f_{\text{clock}} = 3.2 \times 10^{9}\ \text{cycles/s}$

### 1.2 IPC (Instructions Per Cycle)

- **Formula:**
  $$
  \text{IPC} = \frac{\text{Number of retired instructions}}{\text{Number of clock cycles}}
  $$

- **Derived:**
  $$
  \text{Inst/s} = \text{IPC} \times f_{\text{clock}}
  $$

- **Interpretation:** Average completed instructions per clock cycle.


### 1.3 CPI (Cycles Per Instruction)

- **Formula:**
  $$
  \text{CPI} = \frac{\text{Number of clock cycles}}{\text{Number of retired instructions}}
  $$

- **Relation to IPC:**
  $$
  \text{IPC} = \frac{1}{\text{CPI}}, \qquad \text{CPI} = \frac{1}{\text{IPC}}
  $$

### 1.4 FLOPS (Floating-Point Operations Per Second)

#### 1.4.1 Theoretical Peak (Per Device)

- **Formula:**
  $$
  \text{Peak FLOPS} = N_{\text{cores}} \times f_{\text{clock}} \times \text{FLOPs per cycle per core}
  $$

Example (scalar):
- 1 core, 3 GHz, 1 FLOP/cycle => $3 \,\text{GFLOPS}$.

Example (vector FMA):
- 8-wide FMA, 2 FP units/core, 3 GHz:
  $$
  \text{FLOPs/cycle/core} = 8 \times 2 \times 2 = 32
  $$
  $$
  \text{Peak} = 3 \times 10^{9} \times 32 = 96\,\text{GFLOPS}
  $$

#### 1.4.2 Delivered / Measured FLOPS

- **Formula:**
  $$
  \text{Delivered FLOPs/s} = \frac{\text{Floating-point operations actually performed}}{\text{Execution time}}
  $$

- **Efficiency:**
  $$
  \text{FLOP efficiency} = \frac{\text{Delivered FLOPs/s}}{\text{Peak FLOPs/s}} \times 100\%
  $$

### 1.5 MACs (Multiply-Accumulate Operations)

- **Definition:**
  $$
  a \gets a + (b \times c)
  $$

- **MACs per second (per unit):**
  $$
  \text{MAC/s} = N_{\text{MAC/cycle}} \times f_{\text{clock}}
  $$

- **GMAC:**
  $$
  \text{GMAC} = \frac{\text{Number of MAC operations}}{10^{9}}
  $$

- **Relation to FLOPs (when one MAC = 2 FLOPs):**
  $$
  \text{FLOPs} = 2 \times \text{MACs}
  $$

### 1.6 Instruction Mix Ratios

- **FP instruction fraction:**
  $$
  \text{FP fraction} = \frac{\text{FP instructions}}{\text{Total instructions}}
  $$

- **Memory instruction fraction:**
  $$
  \text{Memory inst fraction} = \frac{\text{Load/Store instructions}}{\text{Total instructions}}
  $$

---

## 2. GPU & Accelerator Metrics

### 2.1 Streaming Multiprocessor (SM) Occupancy

$$
\text{Occupancy} = \frac{\text{Active warps per SM}}{\text{Maximum warps per SM}} \times 100\%
$$

- Higher occupancy can hide memory latency through warp switching.
- Limited by register usage, shared memory allocation, and block size.

### 2.2 Warp Execution Efficiency

$$
\text{Warp efficiency} = \frac{\text{Active threads per warp}}{32} \times 100\%
$$

- Divergent branches reduce efficiency (threads in a warp take different paths).

### 2.3 Tensor Core / Matrix Unit Utilization

$$
\text{TC utilization} = \frac{\text{Cycles Tensor Cores are active}}{\text{Total cycles}} \times 100\%
$$

- Tensor Cores perform mixed-precision matrix multiply-accumulate (e.g., FP16 inputs -> FP32 accumulator).
- Peak throughput example (NVIDIA H100 SXM): ~990 TFLOPS (FP16 Tensor), ~1,979 TFLOPS (FP8 Tensor).

### 2.4 GPU Memory Bandwidth Utilization

$$
\text{BW utilization} = \frac{\text{Achieved bandwidth}}{\text{Peak HBM bandwidth}} \times 100\%
$$

- HBM3 example (H100): 3.35 TB/s peak bandwidth.
- HBM3e example (B200): 8 TB/s peak bandwidth.

### 2.5 Kernel Launch Overhead

$$
T_{\text{kernel}} = T_{\text{launch}} + T_{\text{compute}} + T_{\text{sync}}
$$

- Launch overhead: 5-20 us typical on CUDA. Critical for small kernels.
- **Kernel fusion** eliminates intermediate launches and memory round-trips.

### 2.6 GPU Compute vs Memory Bound Classification

| Indicator | Compute Bound | Memory Bound |
|-----------|--------------|--------------|
| SM utilization | High | Low-Medium |
| Memory BW utilization | Low-Medium | High (near peak) |
| Arithmetic Intensity | High | Low |
| Fix strategy | Reduce FLOPs, use lower precision | Improve data reuse, fuse kernels |

### 2.7 NPU / TPU / Custom Accelerator Metrics

- **Systolic array utilization:** fraction of PEs (Processing Elements) active per cycle.
- **On-chip SRAM bandwidth** often exceeds off-chip by 10-100x; data tiling is critical.
- **TPU MXU utilization:**
  $$
  \text{MXU util} = \frac{\text{Performed matmuls}}{\text{Peak matmul capacity}} \times 100\%
  $$

---

## 3. Time, Throughput & Latency

### 3.1 Execution Time (Latency)

- Using CPI:
  $$
  T_{\text{exec}} = \frac{\text{Instruction count} \times \text{CPI}}{f_{\text{clock}}}
  $$

- Using IPC:
  $$
  T_{\text{exec}} = \frac{\text{Instruction count}}{\text{IPC} \times f_{\text{clock}}}
  $$

### 3.2 Throughput

Generic definition:
$$
\text{Throughput} = \frac{\text{Work done}}{\text{Time}}
$$

Special cases:

- Instructions: $\text{Inst/s} = \frac{\text{Instruction count}}{T_{\text{exec}}}$
- FLOPs: $\text{FLOPs/s} = \frac{\text{Floating-point operations}}{T_{\text{exec}}}$
- Requests: $\text{Req/s} = \frac{\text{Number of completed requests}}{T_{\text{obs}}}$

### 3.3 Latency vs Throughput (Simple Case)

For a simple fully pipelined system:
$$
\text{Throughput} \approx \frac{1}{\text{Average latency}}
$$

### 3.4 Tail Latency (Percentile Latency)

- **P50** (median), **P95**, **P99**, **P99.9** latencies.
- Critical for production SLAs. A system can have good average latency but unacceptable tail latency.
- **Rule of thumb:** At scale with fan-out $N$, the probability of hitting a slow node grows as $1 - (1 - p)^N$.

### 3.5 Speedup

Given baseline time $T_1$ and optimized time $T_2$:
$$
S = \frac{T_1}{T_2}
$$

---

## 4. Roofline Model

The roofline model provides a visual upper bound on performance as a function of arithmetic intensity.

### 4.1 Core Equation

$$
\text{Attainable Performance (FLOPs/s)} = \min\!\Big(\text{Peak FLOPs/s},\;\text{Peak BW} \times \text{AI}\Big)
$$

Where:
- $\text{AI}$ = Arithmetic Intensity (FLOPs / Byte)
- $\text{Peak BW}$ = Peak memory bandwidth (Bytes/s)

### 4.2 Ridge Point

The crossover where the model shifts from memory-bound to compute-bound:

$$
\text{AI}_{\text{ridge}} = \frac{\text{Peak FLOPs/s}}{\text{Peak BW}}
$$

- If $\text{AI} < \text{AI}_{\text{ridge}}$ -> **memory-bound** (optimize data movement).
- If $\text{AI} > \text{AI}_{\text{ridge}}$ -> **compute-bound** (optimize arithmetic).

### 4.3 Common AI Workload Arithmetic Intensities

| Operation | Typical AI (FLOPs/Byte) | Bound |
|-----------|------------------------|-------|
| Element-wise (ReLU, add) | 0.25 - 1 | Memory |
| Batch Norm | 1 - 5 | Memory |
| Convolution (large batch) | 10 - 100 | Compute |
| Matrix Multiply (large) | 50 - 200+ | Compute |
| Attention (long seq) | 5 - 50 | Varies |
| Embedding lookup | < 1 | Memory |

### 4.4 Hierarchical Roofline

Extend the roofline with multiple ceilings for different memory levels:
- **L1 cache** roofline -> highest bandwidth, smallest capacity
- **L2 cache** roofline
- **HBM / DRAM** roofline -> lowest bandwidth, largest capacity

Each level adds a bandwidth ceiling; the kernel's performance is bounded by the ceiling corresponding to the memory level it saturates.

---

## 5. Parallel Performance

### 5.1 Amdahl's Law

Let $f$ be the fraction of work that can be accelerated, $s$ the speedup of that fraction:

$$
S = \frac{1}{(1 - f) + \frac{f}{s}}
$$

Special case, parallelization across $N$ cores with ideal scaling ($s = N$):
$$
S_N = \frac{1}{(1 - f) + \frac{f}{N}}
$$

### 5.2 Gustafson's Law

For scaled-size problems on $N$ processors:

$$
S_N^{\text{Gustafson}} = (1 - f) + N \cdot f
$$

### 5.3 Parallel Efficiency

For $N$ workers:

$$
E_N = \frac{S_N}{N}
$$

### 5.4 Load Balance Metric

Let $T_i$ be time spent on worker $i$, and $T_{\max} = \max_i T_i$:

$$
\text{Load balance} = \frac{\left(\sum_i T_i / N\right)}{T_{\max}}
$$

### 5.5 Scaling Efficiency (Weak vs Strong)

- **Strong scaling:** Fixed total problem size, increase $N$.
  $$
  E_{\text{strong}} = \frac{T_1}{N \cdot T_N}
  $$

- **Weak scaling:** Problem size grows with $N$ (constant work per worker).
  $$
  E_{\text{weak}} = \frac{T_1}{T_N}
  $$

---

## 6. Memory & I/O Metrics

### 6.1 Memory Bandwidth

Definition:
$$
\text{Bandwidth} = \frac{\text{Bytes transferred}}{\text{Second}}
$$

Theoretical example (64-bit bus, 3.2 GT/s):
$$
\text{Peak BW} = 8\,\text{bytes} \times 3.2 \times 10^{9}\ \text{transfers/s}
$$

### 6.2 Memory Latency

Conversion between time and cycles:
$$
\text{Latency (cycles)} = \text{Latency (seconds)} \times f_{\text{clock}}
$$

### 6.3 Arithmetic Intensity (AI)

$$
\text{AI} = \frac{\text{Floating-point operations}}{\text{Bytes transferred from memory}}
$$

### 6.4 Cache Hit/Miss Ratios

- Hit ratio:
  $$
  \text{Hit ratio} = \frac{\text{Cache hits}}{\text{Cache accesses}}
  $$

- Miss ratio:
  $$
  \text{Miss ratio} = 1 - \text{Hit ratio} = \frac{\text{Cache misses}}{\text{Cache accesses}}
  $$

### 6.5 Memory Hierarchy Typical Latencies

| Level | Latency | Bandwidth (approx.) |
|-------|---------|---------------------|
| L1 Cache | ~1 ns (3-4 cycles) | ~1-4 TB/s |
| L2 Cache | ~3-10 ns | ~500 GB/s - 1 TB/s |
| L3 Cache | ~10-30 ns | ~200-500 GB/s |
| DRAM (DDR5) | ~50-100 ns | ~50-100 GB/s |
| HBM3 | ~80-120 ns | ~2-4 TB/s |
| NVMe SSD | ~10-100 us | ~5-14 GB/s |
| Network (RDMA) | ~1-5 us | ~25-400 Gb/s |

### 6.6 I/O Throughput

$$
\text{I/O throughput} = \frac{\text{Bytes read or written}}{\text{Second}}
$$

### 6.7 Data Loading Pipeline Throughput

For AI workloads, the training loop is often bottlenecked by data loading:

$$
\text{Data starvation ratio} = \frac{T_{\text{GPU idle waiting for data}}}{T_{\text{total step}}}
$$

- Aim for $\text{Data starvation ratio} \approx 0$ via prefetching, multi-worker data loaders, and on-GPU augmentation.

---

## 7. System-Level & Queueing Metrics

### 7.1 CPU / Device Utilization

Time-based:
$$
\text{Utilization} = \frac{\text{Time device is executing useful work}}{\text{Total observed time}} \times 100\%
$$

### 7.2 Unit Utilization (e.g. FPU, Tensor Core)

$$
\text{Unit utilization} = \frac{\text{Cycles where the unit is busy}}{\text{Total cycles}} \times 100\%
$$

### 7.3 Stall Fraction

$$
\text{Stall fraction} = \frac{\text{Stall cycles}}{\text{Total cycles}}
$$

### 7.4 Little's Law

For a stable system:
$$
L = \lambda \times W
$$

Where:
- $L$ = average number of items in the system (concurrency)
- $\lambda$ = arrival rate (items/s)
- $W$ = average time an item spends in the system (s)

Rearrangements:
$$
\lambda = \frac{L}{W}, \qquad W = \frac{L}{\lambda}
$$

**Applied to inference serving:** If you want $\lambda = 100$ req/s and average latency $W = 50$ ms:
$$
L = 100 \times 0.05 = 5 \text{ concurrent requests in flight}
$$

---

## 8. AI/ML Training Metrics

### 8.1 Training Throughput

$$
\text{Samples/s} = \frac{\text{Batch size}}{T_{\text{step}}}
$$

$$
\text{Total training time} \approx \frac{\text{Total samples (epochs x dataset size)}}{\text{Samples/s}}
$$

### 8.2 Model FLOPs (Forward + Backward)

For a Transformer model with $L$ layers, hidden size $H$, sequence length $S$, vocabulary $V$:

- **Forward pass per token (approx.):**
  $$
  \text{FLOPs}_{\text{fwd}} \approx 2 \times P
  $$
  where $P$ is the number of parameters.

- **Backward pass is approximately 2x forward**, so total per-token training:
  $$
  \text{FLOPs}_{\text{train/token}} \approx 6 \times P
  $$

### 8.3 Hardware FLOPs Utilization (HFU / MFU)

- **Model FLOPs Utilization (MFU):** only counts "useful" model math:
  $$
  \text{MFU} = \frac{\text{Model FLOPs per step} / T_{\text{step}}}{\text{Peak device FLOPs/s}} \times 100\%
  $$

- **Hardware FLOPs Utilization (HFU):** includes all FLOPs the hardware performs (rematerialization, etc.):
  $$
  \text{HFU} = \frac{\text{All FLOPs per step} / T_{\text{step}}}{\text{Peak device FLOPs/s}} \times 100\%
  $$

- Good MFU values: 40-60% (typical), 60%+ (excellent).

### 8.4 Convergence Efficiency

$$
\text{Samples to accuracy} = \text{Total samples processed to reach target metric}
$$

$$
\text{Time to accuracy} = \frac{\text{Samples to accuracy}}{\text{Samples/s}}
$$

- Used by MLPerf Training benchmarks.

### 8.5 Gradient Accumulation Effective Batch Size

$$
\text{Effective batch size} = \text{Micro-batch} \times \text{Accumulation steps} \times N_{\text{GPUs}}
$$

### 8.6 GPU Memory Breakdown (Training)

| Component | Approximate Memory |
|-----------|-------------------|
| Model parameters | $2P$ bytes (FP16) or $4P$ bytes (FP32) |
| Gradients | Same as parameters |
| Optimizer state (Adam) | $8P$-$12P$ bytes (FP32 momentum + variance + master weights) |
| Activations | proportional to batch x seq_len x hidden x layers |
| **Total (FP16 + Adam)** | **~16P-20P bytes** |

---

## 9. AI/ML Inference & Serving Metrics

### 9.1 Inference Latency Breakdown

$$
T_{\text{inference}} = T_{\text{preprocess}} + T_{\text{model}} + T_{\text{postprocess}} + T_{\text{network}}
$$

### 9.2 Batched Throughput vs Latency Trade-off

Increasing batch size typically:
- **Increases throughput** (better hardware utilization, amortized overhead)
- **Increases per-request latency** (queuing delay)

$$
\text{Throughput} = \frac{B}{T_{\text{batch}}(B)}
$$

Optimal batch size maximizes throughput while keeping $T_{\text{batch}}(B) \leq \text{SLA}$.

### 9.3 Model Serving SLA Metrics

| Metric | Definition |
|--------|-----------|
| P50 / P99 latency | 50th / 99th percentile end-to-end latency |
| Throughput (QPS) | Queries per second at target latency |
| Availability | Successful requests / Total requests x 100% |
| Goodput | Throughput of requests meeting SLA |

### 9.4 Dynamic Batching Efficiency

$$
\text{Batch fill rate} = \frac{\text{Average actual batch size}}{\text{Maximum batch size}} \times 100\%
$$

### 9.5 Model Compression Metrics

$$
\text{Compression ratio} = \frac{\text{Original model size}}{\text{Compressed model size}}
$$

$$
\text{Accuracy retention} = \frac{\text{Compressed model accuracy}}{\text{Original model accuracy}} \times 100\%
$$

---

## 10. LLM-Specific Performance Metrics

### 10.1 Time to First Token (TTFT)

$$
\text{TTFT} = T_{\text{receive request}} \to T_{\text{first token generated}}
$$

Includes prompt/prefill processing. Critical for perceived responsiveness.

### 10.2 Time Per Output Token (TPOT)

$$
\text{TPOT} = \frac{T_{\text{generation}} - \text{TTFT}}{\text{Number of output tokens} - 1}
$$

Determines the streaming speed experienced by the user.

### 10.3 Token Throughput

- **Per-request:**
  $$
  \text{Tokens/s (per request)} = \frac{\text{Output tokens}}{T_{\text{generation}}}
  $$

- **System-wide:**
  $$
  \text{Tokens/s (system)} = \frac{\text{Total tokens generated across all requests}}{T_{\text{observation}}}
  $$

### 10.4 Prefill vs Decode Phases

| Phase | Characteristic | Bottleneck |
|-------|---------------|------------|
| **Prefill** | Process all input tokens in parallel | Compute-bound (large matmuls) |
| **Decode** | Generate tokens one at a time (autoregressive) | Memory-bound (KV cache reads, low arithmetic intensity) |

### 10.5 KV Cache Memory

$$
\text{KV cache per token} = 2 \times L \times H \times \text{bytes per element}
$$

$$
\text{KV cache (total)} = \text{KV cache per token} \times S \times B
$$

Where: $L$ = layers, $H$ = hidden dim, $S$ = sequence length, $B$ = batch size.

- For Llama 3 70B (FP16): ~1.3 MB per token -> 1K tokens = ~1.3 GB per request.

### 10.6 LLM Serving Optimization Techniques

| Technique | Effect |
|-----------|--------|
| **Continuous batching** | Dynamically add/remove requests from batch -> higher GPU utilization |
| **PagedAttention (vLLM)** | Non-contiguous KV cache pages -> eliminates memory fragmentation, +2-4x throughput |
| **Speculative decoding** | Draft model proposes tokens, target model verifies in parallel -> lower latency |
| **Prefix caching** | Reuse KV cache for shared prompt prefixes -> faster TTFT for repeated prefixes |
| **Quantized KV cache** | INT8/FP8 KV cache -> 2x more concurrent requests |
| **Flash Attention** | IO-aware exact attention -> reduced memory, faster computation |

### 10.7 LLM Benchmark Metrics

| Benchmark | What it measures |
|-----------|-----------------|
| **MMLU / MMLU-Pro** | Multi-domain knowledge accuracy |
| **HumanEval / MBPP** | Code generation pass@k |
| **MT-Bench** | Multi-turn conversation quality |
| **Chatbot Arena (ELO)** | Human preference ranking |
| **MLPerf Inference** | Standardized latency / throughput |

---

## 11. Distributed Training & Multi-Device Scaling

### 11.1 Data Parallelism

Each worker gets a full model copy and a shard of the data:

$$
\text{Throughput}_{\text{DP}} \approx N \times \text{Throughput}_{\text{single}} \times E_{\text{comm}}
$$

Where $E_{\text{comm}} < 1$ accounts for gradient synchronization overhead.

### 11.2 Communication Overhead (AllReduce)

For ring AllReduce with $N$ nodes and message size $M$:

$$
T_{\text{AllReduce}} \approx 2 \times \frac{N - 1}{N} \times \frac{M}{\text{BW}} + 2(N - 1) \times \alpha
$$

Where $\alpha$ = per-message latency, $\text{BW}$ = per-link bandwidth.

### 11.3 Computation-Communication Overlap Ratio

$$
\text{Overlap ratio} = \frac{\text{Time comm is hidden behind compute}}{\text{Total comm time}}
$$

Ideal: overlap ratio -> 100% (communication fully hidden).

### 11.4 Model Parallelism

**Tensor Parallelism (TP):** Split individual layers across devices.
$$
\text{Comm per layer (TP)} = 2 \times \text{AllReduce}(\text{activation size})
$$

**Pipeline Parallelism (PP):** Assign different layers to different devices.
$$
\text{Pipeline bubble fraction} = \frac{P - 1}{\text{Micro-batches} + P - 1}
$$

Where $P$ = pipeline stages. Larger micro-batch count reduces bubble.

### 11.5 3D Parallelism

$$
N_{\text{total GPUs}} = N_{\text{DP}} \times N_{\text{TP}} \times N_{\text{PP}}
$$

### 11.6 Expert Parallelism (MoE)

$$
\text{All-to-All time} \propto \frac{B \times S \times H \times k}{N_{\text{experts}} \times \text{BW}}
$$

- $k$ = top-k experts per token
- Load imbalance across experts degrades efficiency; auxiliary losses help.

### 11.7 ZeRO (Zero Redundancy Optimizer) Stages

| ZeRO Stage | What is partitioned | Memory saving (per GPU) |
|------------|-------------------|------------------------|
| Stage 1 | Optimizer states | ~4x reduction |
| Stage 2 | + Gradients | ~8x reduction |
| Stage 3 | + Parameters | ~Nx reduction (N = GPUs) |

---

## 12. Quantization & Numerical Precision

### 12.1 Data Types & Their Properties

| Type | Bits | Range (approx.) | AI Use Case |
|------|------|-----------------|-------------|
| FP32 | 32 | +/-3.4x10^38 | Baseline / master weights |
| TF32 | 19 | Same as FP32 (10-bit mantissa) | NVIDIA Ampere+ matmuls |
| BF16 | 16 | +/-3.4x10^38 (7-bit mantissa) | Training (wide range) |
| FP16 | 16 | +/-65,504 (10-bit mantissa) | Training / inference |
| FP8 (E4M3) | 8 | +/-448 | Inference & training (Hopper+) |
| FP8 (E5M2) | 8 | +/-57,344 | Gradients |
| INT8 | 8 | -128 to 127 | Post-training quantization |
| INT4 | 4 | -8 to 7 | Weight-only quantization (GPTQ, AWQ) |
| Binary / Ternary | 1-2 | {-1, 0, 1} | Ultra-low-power edge |

### 12.2 Quantization Speedup Model

$$
\text{Theoretical speedup} \leq \frac{\text{Bits}_{\text{original}}}{\text{Bits}_{\text{quantized}}}
$$

In practice, limited by dequantization overhead, memory alignment, and kernel support.

### 12.3 Quantization-Aware Training (QAT) vs Post-Training Quantization (PTQ)

| Approach | Accuracy | Cost | When to use |
|----------|----------|------|-------------|
| **PTQ** | Good for INT8, degrades at INT4 | Cheap (minutes) | Large models, quick deployment |
| **GPTQ / AWQ** | Good at INT4 weight-only | Moderate (hours) | LLM inference |
| **QAT** | Best accuracy preservation | Expensive (full retrain) | Edge deployment, strict accuracy |

### 12.4 Mixed Precision Training

- Forward: FP16/BF16
- Loss scaling: dynamic to prevent underflow
- Master weights + optimizer: FP32
- Backward: FP16/BF16 gradients

Memory saving: approximately 2x for activations and gradients.

---

## 13. Network & Interconnect Performance

### 13.1 Bandwidth & Latency

$$
T_{\text{transfer}} = \frac{\text{Message size}}{\text{Bandwidth}} + \text{Latency}
$$

### 13.2 Bisection Bandwidth

$$
\text{Bisection BW} = \text{Min aggregate BW across any cut dividing the network in half}
$$

Critical for all-to-all communication patterns.

### 13.3 Common Interconnects for AI

| Interconnect | Bandwidth (per link) | Latency | Topology |
|-------------|---------------------|---------|----------|
| PCIe Gen5 x16 | 64 GB/s | ~100 ns | Point-to-point |
| NVLink 4 (H100) | 900 GB/s (total) | ~1 us | Fully connected (8 GPU) |
| NVLink 5 (B200) | 1.8 TB/s (total) | <1 us | NVLink Switch |
| InfiniBand NDR | 400 Gb/s (50 GB/s) | ~1 us | Fat tree |
| InfiniBand XDR | 800 Gb/s (100 GB/s) | ~1 us | Fat tree / Dragonfly |
| RoCE v2 | 100-400 Gb/s | ~2-5 us | Ethernet fabric |
| Intel Gaudi3 Scale-up | 300 GB/s (per chip) | ~1 us | Mesh |

### 13.4 Network Topology Impact

$$
\text{Comm cost} \propto \text{Hops} \times \frac{\text{Message size}}{\text{Per-hop BW}}
$$

- **Fat-tree**: uniform bisection bandwidth, good for AllReduce.
- **Dragonfly / Torus**: lower cost, but traffic-pattern dependent.

### 13.5 RDMA & GPUDirect

- **GPUDirect RDMA:** GPU memory <-> network without CPU staging -> eliminates copy latency.
- **GPUDirect Storage:** GPU <-> NVMe without CPU -> faster checkpointing.
- **NCCL**: NVIDIA's collective communication library optimized for multi-GPU/multi-node.

---

## 14. Cloud Performance Engineering

### 14.1 Instance Selection Metrics

$$
\text{Cost-efficiency} = \frac{\text{Performance (tokens/s, samples/s, etc.)}}{\text{USD/hour}}
$$

### 14.2 Cloud GPU Instance Comparison (Typical)

| Cloud Instance | GPU | GPU Memory | Interconnect | USD/hr (approx.) |
|---------------|-----|------------|--------------|----------------|
| AWS p5.48xlarge | 8x H100 | 640 GB HBM3 | NVSwitch + EFA | ~60-98 |
| AWS p5e.48xlarge | 8x H200 | 1.13 TB HBM3e | NVSwitch + EFA | ~80-120 |
| GCP a3-megagpu-8g | 8x H100 | 640 GB HBM3 | NVSwitch + GPUDirect | ~60-100 |
| Azure ND H100 v5 | 8x H100 | 640 GB HBM3 | NVSwitch + InfiniBand | ~60-100 |
| AWS inf2.48xlarge | 12x Inferentia2 | 384 GB | NeuronLink | ~12 |

*Prices vary by region, commitment, and spot availability.*

### 14.3 Spot / Preemptible Instance Strategy

$$
\text{Effective cost} = \text{Spot price} \times \frac{T_{\text{total with preemptions}}}{T_{\text{ideal}}}
$$

- Use **checkpointing** to survive preemptions.
- **Checkpoint overhead:** aim for < 5% of step time.

### 14.4 Auto-Scaling Metrics

$$
\text{Scale-up trigger:} \quad \text{Avg queue depth} > \theta_{\text{high}} \;\text{for}\; t > t_{\text{window}}
$$

$$
\text{Cold start latency} = T_{\text{provision}} + T_{\text{model load}} + T_{\text{warmup}}
$$

- Typical cold start: 10s-5min (depending on model size and framework).
- Mitigation: keep-warm policies, pre-built containers, model caching.

### 14.5 Multi-Region & Edge-Cloud Considerations

$$
T_{\text{end-to-end}} = T_{\text{client-to-edge}} + T_{\text{edge-processing}} + T_{\text{edge-to-cloud}} + T_{\text{cloud-processing}}
$$

- **Geo-routing** minimizes $T_{\text{client-to-edge}}$.
- **Model tiering:** small model at edge, large model fallback in cloud.

---

## 15. Edge & Embedded AI Performance

### 15.1 Edge Deployment Constraints

| Constraint | Typical Range |
|-----------|--------------|
| Power budget | 1-30 W (mobile SoC: ~5 W, edge server: ~300 W) |
| Memory | 2-16 GB (shared CPU+GPU) |
| Storage | 32-256 GB eMMC / NVMe |
| Latency SLA | 1-50 ms (real-time) |
| Connectivity | Intermittent or bandwidth-constrained |

### 15.2 Edge AI Performance Metrics

$$
\text{TOPS} = \text{Tera Operations Per Second (INT8 or INT4)}
$$

$$
\text{TOPS/W} = \frac{\text{TOPS}}{\text{Power (watts)}}
$$

| Edge Chip | TOPS (INT8) | TOPS/W | TDP |
|-----------|------------|--------|-----|
| NVIDIA Jetson Orin NX 16GB | 100 | ~5 | 25 W |
| Apple M4 Neural Engine | 38 | ~19 | ~5 W (NE only) |
| Google Coral Edge TPU | 4 | ~2 | 2 W |
| Qualcomm Snapdragon 8 Gen 3 (Hexagon NPU) | 73 | ~15 | ~5 W |
| Intel Meteor Lake NPU | 11 | ~5 | ~10 W |
| Hailo-8L | 13 | ~13 | 1.5 W |

### 15.3 On-Device Optimization Stack

```
Application
    |
Model Optimization (pruning, quantization, distillation)
    |
Model Format (ONNX, TFLite, Core ML, TensorRT, OpenVINO)
    |
Runtime (ONNX Runtime, TFLite, SNPE, QNN, TensorRT)
    |
Hardware (CPU / GPU / NPU / DSP)
```

### 15.4 Real-Time Performance

$$
\text{Real-time feasible} \iff T_{\text{inference}} \leq T_{\text{frame budget}}
$$

At 30 FPS: $T_{\text{frame}} = 33.3$ ms. At 60 FPS: $T_{\text{frame}} = 16.7$ ms.

### 15.5 Model Optimization Techniques for Edge

| Technique | Model Size Reduction | Accuracy Impact | Latency Impact |
|-----------|---------------------|-----------------|----------------|
| INT8 quantization | 4x | < 1% loss (typically) | 2-4x speedup |
| INT4 quantization | 8x | 1-3% loss | 3-6x speedup |
| Structured pruning | 2-10x | 0.5-3% loss | 2-5x speedup |
| Knowledge distillation | N/A (smaller arch) | < 2% loss | Depends on student |
| Neural Architecture Search | Model-dependent | Often better | Optimized for target |

---

## 16. Compiler & Runtime Optimizations

### 16.1 Graph-Level Optimizations

| Optimization | Description | Tools |
|-------------|-------------|-------|
| **Operator fusion** | Merge consecutive ops into one kernel (e.g., Conv+BN+ReLU) | TensorRT, XLA, TVM, torch.compile |
| **Constant folding** | Pre-compute static expressions | All compilers |
| **Dead code elimination** | Remove unused nodes | All compilers |
| **Layout optimization** | NCHW <-> NHWC for target hardware | TensorRT, OneDNN, XNNPACK |
| **Common subexpression elimination** | Reuse identical computations | XLA, Glow |

### 16.2 Kernel-Level Optimizations

| Optimization | Description |
|-------------|-------------|
| **Tiling / Loop blocking** | Fit working set into cache/SRAM |
| **Vectorization** | Use SIMD/SIMT instructions |
| **Loop unrolling** | Reduce loop overhead, enable ILP |
| **Memory coalescing** | Align GPU memory accesses for warp-wide efficiency |
| **Register blocking** | Maximize register reuse in matmuls |

### 16.3 Key AI Compilers & Runtimes

| Tool | Scope | Key Feature |
|------|-------|-------------|
| **torch.compile (Dynamo + Inductor)** | PyTorch graphs | Python-level tracing + Triton codegen |
| **XLA** | TensorFlow / JAX | Whole-program optimization, TPU support |
| **TensorRT** | NVIDIA inference | INT8/FP16 calibration, layer fusion, engine building |
| **TVM / Apache TVM** | Cross-platform | Auto-tuning, BYOC, edge targets |
| **ONNX Runtime** | Cross-platform inference | Execution providers (CUDA, TensorRT, OpenVINO, CoreML) |
| **OpenVINO** | Intel CPUs / GPUs / VPUs | INT8 quantization, model optimizer |
| **Core ML** | Apple Silicon | Neural Engine dispatch, ANE optimization |
| **Triton (OpenAI)** | GPU kernel authoring | Python -> optimized GPU kernels |
| **MLIR** | Compiler infrastructure | Multi-level IR for heterogeneous compilation |

### 16.4 JIT vs AOT Compilation

| Aspect | JIT (Just-In-Time) | AOT (Ahead-Of-Time) |
|--------|----|----|
| Compilation time | Runtime (first run penalty) | Build time |
| Optimization scope | Dynamic shapes, runtime info | Static shapes only |
| Deployment | Requires compiler at runtime | Self-contained binary |
| Use case | Research, dynamic models | Production, edge devices |

---

## 17. Energy, Cost & Sustainability

### 17.1 Power & Energy

- Power:
  $$
  P = \frac{\text{Energy}}{\text{Time}}
  $$

- Energy per operation:
  $$
  E_{\text{op}} = \frac{\text{Energy consumed}}{\text{Number of operations}}
  $$

- Energy efficiency:
  $$
  \text{FLOPs/J} = \frac{\text{FLOPs}}{\text{Energy}}
  $$

### 17.2 Performance per Watt

$$
\text{Perf/W} = \frac{\text{Performance metric}}{\text{Power}}
$$

Examples: GFLOPS/W, Images/s/W, Tokens/s/W

### 17.3 Performance per Dollar

$$
\text{Perf/\$} = \frac{\text{Performance metric}}{\text{Cost}}
$$

### 17.4 Total Cost of Ownership

$$
\text{TCO} = \text{CapEx} + \text{OpEx over lifetime}
$$

Where:
- **CapEx:** Hardware, networking, facility build-out
- **OpEx:** Electricity, cooling, maintenance, staffing, cloud fees

Compare Perf/TCO across options.

### 17.5 AI Training Carbon Footprint

$$
\text{CO}_2\text{e} = \text{Energy (kWh)} \times \text{Carbon intensity (g CO}_2\text{/kWh)}
$$

- Energy = Power x Time
- Carbon intensity varies by region (50-800 g CO2/kWh).
- PUE (Power Usage Effectiveness) of datacenter: 1.1-1.8 typical.

$$
\text{Total energy} = \text{IT energy} \times \text{PUE}
$$

---

## 18. Tooling & Profiling Reference

### 18.1 GPU Profiling

| Tool | Platform | Purpose |
|------|----------|---------|
| `nvidia-smi` | NVIDIA | Real-time GPU utilization, memory, temp, power |
| `nvtop` / `gpustat` | NVIDIA | Interactive GPU monitoring |
| **NVIDIA Nsight Systems** | NVIDIA | System-wide timeline (CPU+GPU+network) |
| **NVIDIA Nsight Compute** | NVIDIA | Kernel-level GPU profiling (occupancy, roofline) |
| **NVIDIA DCGM** | NVIDIA | Datacenter GPU health and metrics |
| `rocm-smi` / `rocprof` | AMD | ROCm GPU profiling |
| **Intel VTune** | Intel | CPU + GPU profiling |
| **AMD Omniperf** | AMD | MI-series kernel profiling |

### 18.2 AI Framework Profiling

| Tool | Framework | Purpose |
|------|-----------|---------|
| `torch.profiler` | PyTorch | Op-level + trace export (with TensorBoard) |
| `torch.cuda.Event` | PyTorch | Precise CUDA timing |
| `jax.profiler` | JAX | XLA HLO trace |
| **TensorBoard Profiler** | TF / PyTorch / JAX | Visual timeline, op stats, memory |
| **Weights & Biases** | Any | Experiment tracking + system metrics |
| **MLflow** | Any | Experiment tracking + model registry |
| **DeepSpeed Flops Profiler** | DeepSpeed | FLOPs counting + communication profiling |

### 18.3 System-Level Profiling (Linux)

| Tool | Purpose |
|------|---------|
| `perf` | CPU performance counters, call graphs |
| `top` / `htop` / `btop` | Process-level CPU, memory overview |
| `vmstat` / `iostat` / `sar` | Memory, I/O, CPU stats |
| `mpstat` | Per-CPU utilization |
| `pidstat` | Per-process stats |
| `strace` / `ltrace` | Syscall / library call tracing |
| `bpftrace` / BCC tools | eBPF-based dynamic tracing |
| `numactl` / `lstopo` | NUMA topology awareness |
| `turbostat` | CPU frequency, C-states, power |
| `pcm` | Intel Performance Counter Monitor |

**Brendan Gregg's 60-second checklist:**
```bash
uptime                  # load averages
dmesg | tail            # kernel errors
vmstat 1                # CPU, memory, I/O
mpstat -P ALL 1         # per-CPU balance
pidstat 1               # per-process CPU
iostat -xz 1            # disk I/O
free -m                 # memory usage
sar -n DEV 1            # network I/O
sar -n TCP,ETCP 1       # TCP stats
top                     # overview
```

### 18.4 Inference Optimization Tools

| Tool | Purpose |
|------|---------|
| **TensorRT** | NVIDIA GPU inference optimization (INT8, FP16 calibration, fusion) |
| **ONNX Runtime** | Cross-platform optimized inference |
| **OpenVINO** | Intel hardware inference optimization |
| **Core ML Tools** | Apple Silicon optimization |
| **TFLite** | Mobile / embedded inference |
| **vLLM** | High-throughput LLM serving (PagedAttention) |
| **TGI** (Text Generation Inference) | HuggingFace LLM serving |
| **SGLang** | Fast LLM serving with RadixAttention |
| **llama.cpp** | CPU/GPU LLM inference (GGUF quantization) |
| **MLC LLM** | Universal LLM deployment (phone, browser, GPU) |

### 18.5 Benchmarking Tools

| Tool | Purpose |
|------|---------|
| **MLPerf** (Training & Inference) | Industry-standard AI benchmarks |
| `sysbench` | CPU, memory, I/O microbenchmarks |
| `fio` | Storage I/O benchmarking |
| `iperf3` | Network bandwidth testing |
| `likwid` | Hardware performance counter toolkit |
| `STREAM` | Memory bandwidth benchmark |
| `HPL` / `HPL-MxP` | LINPACK for HPC / mixed precision |
| **LM Evaluation Harness** | LLM accuracy benchmarks |
| **LLMPerf** (Anyscale) | LLM serving throughput & latency |

---

## 19. Quick Practical Summary

| Goal | Key Metrics & Tools |
|------|-------------------|
| **Model core performance** | IPC/CPI, clock frequency, instruction count |
| **Compute throughput** | FLOPs/s, MACs/s, arithmetic intensity, roofline model |
| **GPU utilization** | SM occupancy, tensor core utilization, memory BW utilization |
| **Training efficiency** | MFU, samples/s, time-to-accuracy, scaling efficiency |
| **Inference latency** | P50/P95/P99 latency, TTFT, TPOT, batching strategy |
| **LLM serving** | Tokens/s, KV cache memory, prefill vs decode bottleneck |
| **Parallelism & scaling** | Amdahl/Gustafson, strong/weak scaling, communication overlap |
| **Memory bottlenecks** | Cache hit ratio, bandwidth utilization, data loading starvation |
| **Quantization** | INT8/INT4 speedup, accuracy retention, compression ratio |
| **Cloud cost efficiency** | Perf/$, spot strategies, auto-scaling, cold start latency |
| **Edge deployment** | TOPS/W, real-time feasibility, on-device optimization stack |
| **System behavior** | Little's Law (concurrency), tail latency, utilization |
| **Energy & sustainability** | FLOPs/J, Perf/W, TCO, CO2e footprint |

---

## 20. References

### Lectures & Courses
- [MIT 6.172 - Performance Engineering of Software Systems (YouTube)](https://www.youtube.com/watch?v=o7h_sYMk_oc&list=PLUl4u3cNGP63VIBQVWguXxZZi0566y7Wf)
  - [MIT 6.172 Labs (GitHub)](https://github.com/simonaertssen/MIT-6.172-Performance-Engineering-of-Software-Systems)
- [Stanford CS149 - Parallel Computing](https://gfxcourses.stanford.edu/cs149)
- [CMU 15-418/618 - Parallel Computer Architecture and Programming](https://www.cs.cmu.edu/~418/)

### Performance Engineering
- [Performance engineering - Wikipedia](https://en.wikipedia.org/wiki/Performance_engineering)
- [Network performance - Wikipedia](https://en.wikipedia.org/wiki/Network_performance)
- [Roofline model - Wikipedia](https://en.wikipedia.org/wiki/Roofline_model)

### Architecture & Hardware
- [Clock rate - Wikipedia](https://en.wikipedia.org/wiki/Clock_rate)
- [Speedup - Wikipedia](https://en.wikipedia.org/wiki/Speedup)
- [Transistor count - Wikipedia](https://en.wikipedia.org/wiki/Transistor_count)
- [Multiply-accumulate operation (MAC) - Wikipedia](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation)
- [Floating point operations per second - Wikipedia](https://en.wikipedia.org/wiki/Floating_point_operations_per_second)
- [Processor (computing) - Wikipedia](https://en.wikipedia.org/wiki/Processor_(computing))
- [Computer performance - Wikipedia](https://en.wikipedia.org/wiki/Computer_performance)
- [Computer performance by orders of magnitude - Wikipedia](https://en.wikipedia.org/wiki/Computer_performance_by_orders_of_magnitude)
- [Hardware acceleration - Wikipedia](https://en.wikipedia.org/wiki/Hardware_acceleration)

### GPU & Accelerators
- [NVIDIA H100 Datasheet](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Google TPU Documentation](https://cloud.google.com/tpu/docs)

### AI/ML Performance
- [MLPerf - ML Commons](https://mlcommons.org/benchmarks/)
- [Efficient Processing of Deep Neural Networks (Sze et al.) - Book](https://www.morganclaypoolpublishers.com/catalog_Orig/product_info.php?products_id=1530)
- [FlashAttention - Dao et al.](https://arxiv.org/abs/2205.14135)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Megatron-LM - NVIDIA](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed - Microsoft](https://github.com/microsoft/DeepSpeed)
- [ZeRO: Memory Optimizations for Training Billion Parameter Models (Rajbhandari et al.)](https://arxiv.org/abs/1910.02054)

### Quantization
- [A Survey of Quantization Methods for Efficient Neural Network Inference (Gholami et al.)](https://arxiv.org/abs/2103.13630)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

### Benchmarking & Profiling
- [Profiling (computer programming) - Wikipedia](https://en.wikipedia.org/wiki/Profiling_(computer_programming))
- [Benchmark (computing) - Wikipedia](https://en.wikipedia.org/wiki/Benchmark_(computing))
- [Software performance testing - Wikipedia](https://en.wikipedia.org/wiki/Software_performance_testing)
- [Analysis of algorithms - Wikipedia](https://en.wikipedia.org/wiki/Analysis_of_algorithms)
- [Best, worst and average case - Wikipedia](https://en.wikipedia.org/wiki/Best,_worst_and_average_case)

### Efficiency & Sustainability
- [Algorithmic efficiency - Wikipedia](https://en.wikipedia.org/wiki/Algorithmic_efficiency)
- [Performance per watt - Wikipedia](https://en.wikipedia.org/wiki/Performance_per_watt)
- [Green computing - Wikipedia](https://en.wikipedia.org/wiki/Green_computing)
- [Environmental impact of artificial intelligence - Wikipedia](https://en.wikipedia.org/wiki/Environmental_impact_of_artificial_intelligence)

### Compilers & Optimization
- [Program optimization - Wikipedia](https://en.wikipedia.org/wiki/Program_optimization)
- [Optimizing compiler - Wikipedia](https://en.wikipedia.org/wiki/Optimizing_compiler)
- [torch.compile Documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [XLA - Accelerated Linear Algebra](https://openxla.org/xla)
- [Apache TVM](https://tvm.apache.org/)
- [Triton Language (OpenAI)](https://triton-lang.org/)
- [MLIR - Multi-Level IR Compiler Framework](https://mlir.llvm.org/)

### Supercomputing
- [TOP500 - Wikipedia](https://en.wikipedia.org/wiki/TOP500)
- [Supercomputer - Wikipedia](https://en.wikipedia.org/wiki/Supercomputer)

### Linux Performance
- [Linux Performance Analysis in 60,000 Milliseconds - Brendan Gregg @ Netflix](https://netflixtechblog.com/linux-performance-analysis-in-60-000-milliseconds-accc10403c55)
- [Linux Systems Performance - Brendan Gregg](https://www.brendangregg.com/linuxperf.html)
- [Top 10 ways to monitor Linux in the console - Jeff Geerling](https://www.jeffgeerling.com/blog/2025/top-10-ways-monitor-linux-console)

### Blog Posts & Articles
- [AI Performance Engineering (2025-2026 Edition): Latency, Throughput, Cost Optimization & Real-World Benchmarking - Robi Kumar Tomar](https://medium.com/@robi.tomar72/ai-performance-engineering-2025-2026-edition-latency-throughput-cost-optimization-142eec0daece)
- [Efficiently Scaling Transformer Inference - Pope et al. (Google)](https://arxiv.org/abs/2211.05102)
- [LLM Inference Performance Engineering: Best Practices - NVIDIA Technical Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

### Books
- [High Performance Computing - cs-books collection](https://github.com/afondiel/cs-books?tab=readme-ov-file#%EF%B8%8F-computer-architecture)
- [Computer Architecture: A Quantitative Approach - Hennessy & Patterson](https://dl.acm.org/doi/book/10.5555/1999263)
- [Programming Massively Parallel Processors - Kirk & Hwu](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)

---

> *"It's hardware that makes a machine fast. It's software that makes a fast machine slow."* - Craig Bruce

