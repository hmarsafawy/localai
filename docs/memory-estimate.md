# How the app estimates GPU memory

Whether a model fits the GPU decides most of what the app recommends: which
quantization to download, how long a context to run, and whether to expect it
to spill into system RAM. Until now that was `file size × 1.2` against free
VRAM. That's wrong in both directions. It ignores the KV cache, which grows
with the context and differs tenfold between models of the same size. It also
charges a fixed 20% that has nothing to do with what llama.cpp actually
allocates.

`services/memory_estimate.py` replaces it with the sum of what llama-server
allocates, term by term, from the model's shape (`services/model_shape.py`,
read from the GGUF's metadata or a `config.json`). Each term below says what
llama.cpp keeps, why, the formula, and how close the formula came to
llama-server's own numbers on a real GPU.

## The model

```
total = weights + KV cache + recurrent state + compute buffer
        + vision encoder + GPU runtime

fits  = total + 1 GiB margin <= free VRAM
```

All of it depends on how llama-server runs, so the estimate assumes its
defaults (`RuntimeTraits`):

| Setting | Default | Why it matters |
|---|---|---|
| Slots (`--parallel`) | 4 | Each slot keeps its own recurrent state and its own sliding window. |
| Unified KV cache | on | The 4 slots share one cache of `context` cells, so any one conversation can use all of it. |
| `n_ubatch` | 512 | Prompts are processed 512 tokens at a time; the compute buffer is sized for that. |
| KV cache type | f16 | 2 bytes per value. `q8_0` would be 34 bytes per 32 values. |
| Flash attention | on | Without it, attention scores take memory that grows with the context. |

## Weights

**What:** the tensors that go on the GPU.

**From a listing:** only the GGUF's size is known, so the weights are the file
size, minus the layers of a multi-token-prediction (MTP) head. llama.cpp
doesn't load those (its log says `model has unused tensor blk.32.*`). Their
share is taken off assuming every layer is the same size: Qwen3.5-9B MTP has 33
stored layers and runs 32, so 1/33 comes off.

**What this gets wrong:** llama.cpp keeps the token embeddings
(`vocab × n_embd`) on the CPU. If the model's output layer reuses them, a copy
goes on the GPU anyway, but a separate output layer leaves them CPU-only, and
metadata doesn't say which kind a model is. So the estimate counts them,
which only ever overstates. The two models measured with a separate output
layer show how much: 546 MiB for Qwen3.5-9B Q4_K_M and 417 MiB for IQ2_XXS. The
tensor list in the file settles it, and the launch check (PR 4) will read it.

## KV cache

**What:** for every token in the context, each attention layer keeps its key
and value vectors so later tokens can attend to them. This is the term that
grows with the context.

**Formula:** for each attention layer,

```
cells × n_head_kv × (key_length + value_length) × 2 bytes
```

`cells` is the context rounded up to a multiple of 256, as llama.cpp pads it.
Layers with no KV heads (recurrent ones) keep nothing.

**Examples:**

- **Llama 3 8B:** 32 layers × 8 KV heads × (128 + 128) × 2 B = **128 KiB per
  token**, so 2 GiB at 16K and 16 GiB at 128K. It has 32 query heads, but only
  the 8 KV heads are cached (grouped-query attention).
- **Qwen3.5-9B:** only 8 of its 32 layers have attention. 8 × 4 × 512 × 2 B =
  32 KiB per token, so 512 MiB at 16K.
- **DeepSeek V3 (MLA):** each layer caches one compressed latent of 576 values
  and reads V back out of it, so 61 × 576 × 2 B = 69 KiB per token. Counted as
  128 heads of K and V, it would be 71 times that.

**Measured:** exact in every run below.

## Sliding-window layers

**What:** some layers only attend to the last `window` tokens. Gemma 3 does
this in 5 of every 6 layers, with a 1024-token window. llama.cpp gives those
layers a smaller cache.

**Formula:**

```
cells = min(context, pad256(window × slots + n_ubatch))
```

Every slot can be at a different point in its own conversation, so each needs
its own window, plus room for the batch being added.

**Example:** Gemma 3 12B at 16K. 8 full layers × 16,384 cells plus 40 window
layers × 4,608 cells, at 8 KiB per cell, comes to 2,464 MiB instead of 6 GiB.
At 128K it's 9.4 GiB instead of 48.

**Measured:** LFM3-1.7B has a 256-token window in 5 layers:
`pad256(256 × 4 + 512)` = 1,536 cells, so 15 MiB, which is exactly what
llama-server allocated.

## Recurrent state

**What:** Mamba layers, gated delta nets (Qwen3.5, Qwen3-Next), LFM's short
convolutions and RWKV don't keep a cache that grows. They keep a fixed state
per sequence instead: the last few inputs of a convolution, plus a state
matrix. llama-server keeps one per slot.

**Formula:**

```
recurrent layers × state values × 4 bytes (f32) × slots
```

The state values follow llama.cpp's `n_embd_r() + n_embd_s()`:

- **Mamba and delta nets:** `(conv_kernel − 1) × (inner + 2 × groups × state)
  + state × inner`.
- **LFM:** `n_embd × (l_cache − 1)`.
- **RWKV:** `token_shift × n_embd + n_embd × head_size`.

**Example:** Qwen3.5-9B: 24 layers × 548,864 values × 4 B × 4 slots =
**201 MiB**, the same at 4K or 256K.

**Measured:** 201.00 MiB on llama-server, and 50.25 MiB on llama-fit-params,
which runs one slot.

## Compute buffer

**What:** scratch space for one forward pass. llama.cpp reserves it for the
worst case up front: a full batch of 512 tokens. Its size is the most memory
alive at any one point in the graph, which comes down to a few big tensors:

1. **Attention masks:** one per kind of attention layer (full and
   sliding-window), `cells × 512`. They're f16 with flash attention and f32
   without, and they grow with the context: 16 MiB at 16K with flash attention.
2. **Logits:** llama-server reserves them for one token per slot, so
   `vocab × 4 × 4 B`: 3.8 MiB for Qwen3.5's 248K vocabulary.
   (llama-fit-params reserves them for all 512 tokens, which is 485 MiB; see
   the notes below.)
3. **The bigger of two peaks:**
   - **The feed-forward:** gate, up projection and their product, all alive at
     once, `3 × n_ff` values per token. For a mixture of experts,
     `experts_used × (3 × expert_ff + n_embd)`, because each chosen expert's
     output is held apart until they're summed.
   - **Attention scores, without flash attention only:**
     `n_head × cells × 512` f32 values, the whole score matrix. This grows
     with the context and dominates once it's long: 512 MiB for Qwen3.5 at
     16K, 2 GiB at 64K. With flash attention it's never materialised.

   Each of these comes with a few f32 copies of the hidden state: the
   residual stream, its normed copy, the layer's output. That's 4 widths
   alongside the feed-forward and 7 alongside attention scores. These counts
   were fitted to llama-server's buffers and rounded up.

**Formula:**

```
masks + logits + max(
    (3 × n_ff + 4 × n_embd) × 512 × 4 B,
    (n_head × cells + 7 × n_embd) × 512 × 4 B     ← flash attention off only
)
```

A model with no feed-forward at all (Mamba, RWKV) is given `4 × n_embd` in its
place, which is what Mamba's input projection produces.

**Example:** Qwen3.5-9B at 16K comes to 16 MiB of mask, 3.8 MiB of logits and
104 MiB for the feed-forward with its hidden-state copies, so
**123.8 MiB**; llama-server allocated 120.0
MiB. With flash attention off, the scores win at 568 MiB, for 603.8 MiB in
all; llama-server allocated 596.0 MiB.

**Measured:** never under, and at most 12 MiB over, in all 12 runs.

## Vision encoder

**What:** the encoder's weights, plus its own compute buffer, plus memory the
GPU runtime keeps after processing an image.

**Formula:** `encoder file + 768 MiB`.

**Measured:** Qwen3.5-9B's 876 MiB encoder took **1,580 MiB** on ROCm:

- a 248 MiB compute buffer;
- 456 MiB the runtime kept after llama.cpp warmed the encoder up on a
  worst-case 1472×1472 image.

An older Vulkan build, which doesn't do that warmup, took 1,127 MiB. The
constant was 512 MiB until now, measured on an older build. That's 192 MiB
short on the current one, so launches with an encoder now leave 256 MiB more
room.

## GPU runtime

**What:** the GPU runtime's own memory in llama-server's process: the driver
context, compiled kernels, and scratch memory that large matrix products
leave cached. llama.cpp's buffer sizes never include it, but the VRAM we read
before launching has to cover it.

**Measured:** process VRAM minus llama.cpp's buffers, on the RX 9070 XT:

| Backend | Right after loading | After a 6,390-token prompt | Constant |
|---|---|---|---|
| ROCm (b12359) | 271–279 MiB | 275–335 MiB | 384 MiB |
| Vulkan (b10878) | 81–87 MiB | 111–128 MiB | 192 MiB |
| CUDA, Metal | not measured | | 384 MiB (CUDA shares ROCm's code) |

This is also why the estimate's longest context comes out a little shorter
than llama.cpp's own `--fit`, which counts only its own buffers (see below).

## Fit margin

llama.cpp's `--fit` leaves 1 GiB free by default (`--fit-target`), and the app
passes that to every launch. So a model is rated as fitting when
`total + 1 GiB` is free. The margin also absorbs whatever else grows on the
GPU meanwhile, so Windows doesn't start moving memory into system RAM.

## The longest context that fits

`max_gpu_context` finds the longest context that keeps everything on the GPU.
Memory only grows with the context, so it bisects over multiples of 256:

- between 4,096 (`--fit`'s floor; below that, it moves layers off instead) and
  the context the model was trained for;
- capped at 16K if the model doesn't say what it was trained for;
- `None` if even 4,096 doesn't fit.

**Checked against llama-server.** Qwen3.5-9B MTP Q4_K_M with its encoder, with
12,800 MiB of VRAM actually free:

- **`--fit` chose 119,552 tokens.** Windows then placed 479 MiB of the
  process's GPU memory in system RAM. About 250 MiB of that is more than its
  host buffers explain, so it was already past the edge.
- **`max_gpu_context` gives 108,032** with the weights llama.cpp actually
  loaded. That's 11,520 tokens shorter: at 33 KiB per token, the 384 MiB the
  runtime takes.
- **From the file size alone it gives 92,416**, because the 546 MiB of
  embeddings that stay on the CPU are counted.

## The target context

A model's rating is for one context: **16,384 tokens**, or the context the
model was trained for if that's shorter. That's enough for a long conversation
or a few files, and it's the setting where the KV cache is what tells
same-sized models apart.

## The measurements

These runs were on an RX 9070 XT with 16 GB, using llama.cpp b12359 (ROCm) and
b10878 (Vulkan). The buffer sizes are from llama-server's log at `-lv 4`. VRAM
is the process's dedicated GPU memory from Windows' `GPU Process Memory`
counters, after a 6,390-token prompt (\* = right after loading; a prompt adds
5–58 MiB). The totals use the weights llama.cpp actually loaded, to test the
memory model itself; the error from file sizes is covered under Weights.

Each cell is estimate / measured, in MiB.

| Model, context | KV + state | Compute | Total / VRAM |
|---|---|---|---|
| SmolLM2-135M, 8K | 180 / 180 | 22.2 / 21.5 | 685 / 575 |
| SmolLM2-135M, 16K, no flash attention | 360 / 360 | 328.6 / 326.4 | 1,171 / 1,056\* |
| LFM3-1.7B, 16K | 209.4 / 209.4 | 80.5 / 78.7 | 2,427 / 2,312\* |
| LFM3-1.7B, 64K | 785.4 / 785.4 | 128.5 / 126.7 | 3,051 / 2,937\* |
| LFM3-1.7B, 16K, no flash attention | 209.4 / 209.4 | 577.0 / 571.0 | 2,923 / 2,805\* |
| Qwen3.5-9B IQ2_XXS, 16K | 713 / 713 | 123.8 / 120.0 | 3,837 / 3,782 |
| Qwen3.5-9B IQ2_XXS, 64K | 2,249 / 2,249 | 171.8 / 168.0 | 5,421 / 5,312\* |
| Qwen3.5-9B IQ2_XXS, 16K, no flash attention | 713 / 713 | 603.8 / 596.0 | 4,317 / 4,202\* |
| Qwen3.5-9B IQ2_XXS + encoder, 16K | 713 / 713 | 123.8 / 120.0 | 5,480 / 5,362 |
| Qwen3.5-9B MTP Q4_K_M + encoder, 16K | 713 / 713 | 123.8 / 120.0 | 8,821 / 8,652\* |
| SmolLM2-135M, 16K, Vulkan | 360 / 360 | 30.2 / 27.3 | 681 / 597 |
| Qwen3.5-9B IQ2_XXS, 16K, Vulkan | 713 / 713 | 123.8 / 112.0 | 3,645 / 3,569 |

Every total is above what was measured, by 55–169 MiB. That's mostly the
runtime constant's headroom, which is what keeps the estimate on the safe side
for models bigger than these.

## Where it can be wrong

**Overstates:**

- Token embeddings that stay on the CPU, from file sizes: up to ~0.5 GiB on
  models with big vocabularies.
- The GGUF header (the tokenizer) is counted as weights: a few MiB.
- The MTP share assumes MTP layers are average-sized.
- The vision encoder constant is for a large encoder (an 876 MiB file); a
  small one needs less.
- Unknown sliding-window patterns are treated as full attention.

**Could understate:**

- If llama.cpp turns flash attention off for a model, it needs
  `n_head × context × 2 KiB` more. It does that when a backend lacks the kernel
  for the model's head size. Not seen on ROCm or Vulkan; the launch check
  will catch it.
- The runtime overhead on CUDA and Metal is assumed, not measured.
- Mixture-of-experts compute follows llama.cpp's graph but hasn't been
  measured on a real MoE model yet.
- The vision overhead was measured with one encoder.

## Notes from measuring

**The flash-attention warning:** "Flash Attention over K/V rows not supported,
set to disabled" doesn't mean flash attention is off. That's one variant of
it; llama-server logs "Flash Attention enabled" just before. The compute buffer
grows by exactly the f16 mask (1 KiB per token), not by the attention scores.

**llama-fit-params isn't configured like llama-server.** It runs one slot
with no unified cache, and reserves logits for all 512 tokens. So its numbers
differ from what llama-server will allocate:

- Qwen3.5: 485 MiB too high for logits, 150 MiB too low for state.
- Gemma 3 12B: by the window formula, 960 MiB too low for its sliding
  windows.

The estimate follows llama-server.

## Re-measuring

To check a term, or to fill in a backend that isn't measured:

1. Run llama-server with `-lv 4`. It logs each buffer it allocates: `model
   buffer size`, `KV buffer size`, `RS buffer size`, `compute buffer size`,
   and `mtmd_context: ... compute buffer size` for the encoder.
2. Once the model has loaded, and again after a long prompt, read the
   process's VRAM. On Windows that's `loaded.gpu_memory` in
   `GET /servers/{id}`, read from the `GPU Process Memory` counters right after
   loading.
3. VRAM minus the buffers is the runtime's share.
