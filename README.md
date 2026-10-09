# Attention, in order

An interactive, chronological history of attention mechanisms, from 2014's additive attention to DeepSeek-V4's compressed sparse attention. Built as ERA V5 Session 8 assignment.

The site covers 29 mechanisms in the order they were published. Each one is explained as an answer to a problem that existed at that moment, with what it buys, what it costs, and when I would pick it. Every date is checked against a primary source and stored in one machine-readable ledger.

## What is on the page

1. **The attention engine.** One causal attention head computed live on hand-made 8-number embeddings. Pick a sentence (or type one), click any word to make it the query, and watch the five stages: project into query/key/value, the query visiting every key with its dot product, softmax competition, values flowing back in proportion to the weights, and the residual update moving the word on a 2-D meaning map ("bank" moves towards *nature* next to "river", towards *money* next to "cash"). Mask, temperature and a second "previous-token" head are live controls. It is a readable toy with real arithmetic, not a trained model.
2. **The same computation as a matrix**, every number visible, and **the two bills**: T² compute versus the per-user KV cache.
3. **The map.** Every mechanism on one time axis, in the row of the bill it mainly attacks (compute, memory, position, fidelity), with a dashed tail to its first notable use in a released model.
4. **The timeline.** 29 mechanisms in strict date order, grouped only by time into eras. Each card has the moment (what existed, what hurt), what changed, an interactive figure, what it buys, what it costs, when to pick it, what came next, and an expandable date-and-sources panel. A sticky rail shows the year and which bills the mechanism relieves or pays into.
5. **What the order shows**, **workload advice** (2K chatbot vs 128K documents vs 1M agent), **checks on the notes I learned from**, and the **method and sources** table.

## What the timeline shows

The analysis on the page is computed from the ledger. In short:

- **Ideas wait for their bill.** Multi-query attention, top-k selection, sliding windows, linear attention and the delta rule all appeared in 2019–2021, and each waited years for a flagship model. Grouped-query attention was in Llama 2 two months after publication. The ideas did not change; the cost that hurt (the per-user KV cache) did.
- **Compute came first.** The 2019–2020 wave was mostly about compute. FlashAttention (May 2022) made exact attention fast in practice, and only after that did memory (GQA, MLA) and length (PI, NTK-aware scaling, YaRN, all within about ten weeks in 2023) take over.
- **The field went back to a fixed-size state.** Attention was invented to escape one. Linear attention reintroduced it for cost, and the next five years made that state smarter (delta rule, Mamba, Gated DeltaNet). Production models that use fixed state all keep some exact attention layers.
- **Position swings like a pendulum**, from learned tables to sinusoids, relative terms, RoPE and ALiBi, NoPE, stretched RoPE, and finally removing RoPE after training (DroPE).
- **Old ideas come back with their missing piece.** Top-k selection (2019) still scored every key; it returned in DeepSeek's NSA and DSA (2025) with a cheap, trained selector.

## Run it

Requires Node 20+ (tested with Node 22).

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check, ledger/README audit, production build into dist/
npm run preview    # serve dist/ on http://localhost:4173
npm test           # maths tests: softmax, RoPE relativity, linear-attention equivalence, delta rule, KV formula, YaRN/NTK/PI
npm run qa         # browser QA against a running preview (needs Chrome installed)
```

`dist/` is a static site that any static host can serve. `netlify.toml` holds the Netlify build settings.

## How the dates were decided

**Rule.** Each entry's date is the earliest publicly verifiable appearance of the primary source that introduced the technique in the form, and under the name, the field now uses. For arXiv papers that is the **v1 "Submitted on" date** on the abstract page (not the later conference date). For forum posts and releases it is the first-party post or announcement date. Earlier work that computes substantially the same operation in another setting (an RNN, a memory network) is shown on the card as an **earlier root** with its own verified date, rather than silently moving the date.

**Process.** A first research pass fetched every primary page (arXiv abstract pages, PDFs, official release pages) and recorded the evidence. An independent second pass re-fetched every source and looked for earlier appearances. The research passes were LLM-assisted, and I reviewed the results. The second pass changed things: learned absolute position embeddings moved back six months (Gehring et al., Nov 2016, an earlier paper by the ConvS2S group); earlier roots were added for self-attention, relative distance terms, top-k selection, sliding windows, ALiBi (the T5 bias) and hybrids; BLOOM's adoption date was corrected to its model release; the DeepSeek-V4 report's arXiv listing was confirmed; and unverifiable claims (an exact date for Jianlin Su's original RoPE blog, GPT-J's release month) were removed or downgraded.

**Single source of truth.** Every date on the page and in the tables below is read from `src/data/ledger.json`. `scripts/check-ledger.mjs` fails the build if dates are malformed, a precursor is dated after its entry, a card has no ledger entry, or a year written in the hand-written text is not backed by a ledger date. `scripts/gen-sources.mjs --check` fails the build if the tables below drift from the ledger.

**Known soft spots.** The NTK-aware Reddit post's timestamp was read from the Pullpush archive because reddit.com blocked automated fetching (the YaRN paper cites the same post). The exact date of Jianlin Su's Chinese-language RoPE blog posts (early 2021) could not be fetched, so the arXiv paper sets the date. GPT-1's release date is as reported by search because openai.com blocked automated fetching. The DeepSeek-V4 report has the arXiv ID 2606.19348, but its abstract page states v1 was submitted on 26 April 2026; the timeline uses the release announcement (24 April 2026). None of these change the order of the timeline.

<!-- sources:start (generated by scripts/gen-sources.mjs from src/data/ledger.json — do not edit by hand) -->
### Timeline dates and their primary sources

| Date | Mechanism | What the date is | Primary source |
|---|---|---|---|
| 1 Sep 2014 | Additive attention for translation *(added)* | arXiv v1 of Bahdanau, Cho & Bengio | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Bahdanau, Cho, Bengio (arXiv:1409.0473) |
| 7 Nov 2016 | Learned absolute position embeddings | arXiv v1 of Gehring et al., “A Convolutional Encoder Model for Neural Machine Translation”, which adds learned position embeddings for the absolute position of each source word | [A Convolutional Encoder Model for Neural Machine Translation](https://arxiv.org/abs/1611.02344) — Gehring, Auli, Grangier, Dauphin (arXiv:1611.02344) |
| 12 Jun 2017 | Scaled dot-product attention (with multi-head and the causal mask) | arXiv v1 of 'Attention Is All You Need' | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (arXiv:1706.03762) |
| 12 Jun 2017 | Sinusoidal position encoding | arXiv v1 of 'Attention Is All You Need' (same paper as scaled dot-product attention) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (arXiv:1706.03762) |
| 6 Mar 2018 | Relative position representations *(added)* | arXiv v1 of Shaw, Uszkoreit & Vaswani | [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) — Shaw, Uszkoreit, Vaswani (arXiv:1803.02155) |
| 9 Jan 2019 | Segment recurrence (Transformer-XL) *(added)* | arXiv v1 of Dai et al. | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) — Dai, Yang, Yang, Carbonell, Le, Salakhutdinov (arXiv:1901.02860) |
| 23 Apr 2019 | Fixed sparse attention patterns (Sparse Transformer) | arXiv v1 of Child, Gray, Radford & Sutskever | [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — Child, Gray, Radford, Sutskever (arXiv:1904.10509) |
| 6 Nov 2019 | Multi-query attention (MQA) | arXiv v1 of Shazeer, 'One Write-Head is All You Need' | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — Shazeer (arXiv:1911.02150) |
| 25 Dec 2019 | Top-k attention (Explicit Sparse Transformer) | arXiv v1 of Zhao et al. | [Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection](https://arxiv.org/abs/1912.11637) — Zhao, Lin, Zhang, Ren, Su, Sun (arXiv:1912.11637) |
| 10 Apr 2020 | Sliding-window attention | arXiv v1 of Longformer, which names and studies 'sliding window attention' (plus dilated and global variants) | [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy, Peters, Cohan (arXiv:2004.05150) |
| 29 Jun 2020 | Linear attention | arXiv v1 of Katharopoulos et al., 'Transformers are RNNs' | [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) — Katharopoulos, Vyas, Pappas, Fleuret (arXiv:2006.16236) |
| 22 Feb 2021 | The delta rule for linear attention | arXiv v1 of Schlag, Irie & Schmidhuber, which introduces the delta-rule update for linear Transformers | [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) — Schlag, Irie, Schmidhuber (arXiv:2102.11174) |
| 20 Apr 2021 | Rotary position embedding (RoPE) | arXiv v1 of RoFormer (Su et al.) | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su, Lu, Pan, Murtadha, Wen, Liu (arXiv:2104.09864) |
| 27 Aug 2021 | ALiBi (attention with linear biases) | arXiv v1 of Press, Smith & Lewis, 'Train Short, Test Long' | [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — Press, Smith, Lewis (arXiv:2108.12409) |
| 30 Mar 2022 | No positional encoding at all (NoPE) *(added)* | arXiv v1 of Haviv et al., showing causal LMs without positional encodings stay competitive | [Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634) — Haviv, Ram, Press, Izsak, Levy (arXiv:2203.16634) |
| 27 May 2022 | FlashAttention (IO-aware exact attention) *(added)* | arXiv v1 of Dao et al. | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao, Fu, Ermon, Rudra, Ré (arXiv:2205.14135) |
| 28 Dec 2022 | Hybrid stacks: mostly state layers, a few attention layers *(added)* | arXiv v1 of H3, whose hybrid model keeps two attention layers among state-space layers | [Hungry Hungry Hippos: Towards Language Modeling with State Space Models](https://arxiv.org/abs/2212.14052) — Fu, Dao, Saab, Thomas, Rudra, Ré (arXiv:2212.14052) |
| 22 May 2023 | Grouped-query attention (GQA) | arXiv v1 of Ainslie et al. | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (arXiv:2305.13245) |
| 27 Jun 2023 | Position Interpolation *(added)* | arXiv v1 of Chen et al. (Meta) | [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Chen, Wong, Chen, Tian (arXiv:2306.15595) |
| 29 Jun 2023 | NTK-aware RoPE scaling | Reddit post by u/bloc97 on r/LocalLLaMA (08:21 UTC) | [NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) — bloc97 (Reddit) (reddit:14lz7j5) |
| 31 Aug 2023 | YaRN | arXiv v1 of Peng, Quesnelle, Fan & Shippole | [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — Peng, Quesnelle, Fan, Shippole (arXiv:2309.00071) |
| 29 Sep 2023 | Attention sinks (StreamingLLM) | arXiv v1 of Xiao et al., 'Efficient Streaming Language Models with Attention Sinks' | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — Xiao, Tian, Chen, Han, Lewis (arXiv:2309.17453) |
| 1 Dec 2023 | Selective state spaces (Mamba) *(added)* | arXiv v1 of Gu & Dao | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu, Dao (arXiv:2312.00752) |
| 7 May 2024 | Multi-head latent attention (MLA) | arXiv v1 of the DeepSeek-V2 technical report | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek-AI (arXiv:2405.04434) |
| 9 Dec 2024 | Gated DeltaNet | arXiv v1 of Yang, Kautz & Hatamizadeh | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — Yang, Kautz, Hatamizadeh (arXiv:2412.06464) |
| 16 Feb 2025 | Native Sparse Attention (DeepSeek) | arXiv v1 of Yuan et al. (DeepSeek-AI & Peking University) | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) — Yuan et al. (DeepSeek-AI, PKU) (arXiv:2502.11089) |
| 29 Sep 2025 | DeepSeek Sparse Attention (lightning indexer + top-k) | DeepSeek-V3.2-Exp release announcement (DeepSeek API docs news) | [Introducing DeepSeek-V3.2-Exp](https://api-docs.deepseek.com/news/news250929) — DeepSeek-AI (deepseek-news-250929) |
| 13 Dec 2025 | DroPE (dropping positional embeddings after pretraining) | arXiv v1 of Gelberg, Eguchi, Akiba & Cetin (Sakana AI) | [Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings](https://arxiv.org/abs/2512.12167) — Gelberg, Eguchi, Akiba, Cetin (arXiv:2512.12167) |
| 24 Apr 2026 | Compressed Sparse + Heavily Compressed Attention (DeepSeek-V4) | DeepSeek-V4 Preview release announcement (DeepSeek API docs news) | [DeepSeek-V4 Preview release](https://api-docs.deepseek.com/news/news260424) — DeepSeek-AI (deepseek-news-260424) |

### Earlier roots shown on the cards

| Mechanism | Earlier root | Date | Source |
|---|---|---|---|
| Additive attention | Graves, 'Generating Sequences With Recurrent Neural Networks' — a soft, differentiable window over the input for handwriting synthesis | 4 Aug 2013 | [link](https://arxiv.org/abs/1308.0850) |
| Learned positions | End-To-End Memory Networks (Sukhbaatar et al.) — learned 'temporal encoding' rows indexed by memory slot | 31 Mar 2015 | [link](https://arxiv.org/abs/1503.08895) |
| Transformer attention | Additive attention over RNN states (Bahdanau et al.) | 1 Sep 2014 | [link](https://arxiv.org/abs/1409.0473) |
| Transformer attention | Luong et al. — dot-product ('multiplicative') attention scores | 17 Aug 2015 | [link](https://arxiv.org/abs/1508.04025) |
| Transformer attention | Parikh et al. — intra-sentence (self-)attention | 6 Jun 2016 | [link](https://arxiv.org/abs/1606.01933) |
| Sinusoidal | End-To-End Memory Networks: a fixed, computed 'position encoding' for word order inside a sentence | 31 Mar 2015 | [link](https://arxiv.org/abs/1503.08895) |
| Relative positions | Parikh et al. — distance-sensitive bias terms inside intra-sentence attention | 6 Jun 2016 | [link](https://arxiv.org/abs/1606.01933) |
| Sparse Transformer | Generating Wikipedia by Summarizing Long Sequences — local attention blocks and memory-compressed attention | 30 Jan 2018 | [link](https://arxiv.org/abs/1801.10198) |
| Sparse Transformer | Image Transformer — blocks of local attention | 15 Feb 2018 | [link](https://arxiv.org/abs/1802.05751) |
| Top-k attention | Rae et al., Sparse Access Memory — reads only the K nearest memory words | 27 Oct 2016 | [link](https://arxiv.org/abs/1610.09027) |
| Top-k attention | Sparse Attentive Backtracking (Ke et al.) — 'selects and normalizes only the k top greatest raw attention weights' over past RNN states | 11 Sep 2018 | [link](https://arxiv.org/abs/1809.03702) |
| Sliding window | Chorowski et al. — attention restricted to a window around the previous alignment (speech, RNN) | 24 Jun 2015 | [link](https://arxiv.org/abs/1506.07503) |
| Sliding window | Luong et al. — 'local attention' over a window around an aligned position (RNN NMT) | 17 Aug 2015 | [link](https://arxiv.org/abs/1508.04025) |
| Sliding window | Liu et al. — local attention within non-overlapping blocks | 30 Jan 2018 | [link](https://arxiv.org/abs/1801.10198) |
| Sliding window | Image Transformer (Parmar et al.) — local 1D/2D attention | 15 Feb 2018 | [link](https://arxiv.org/abs/1802.05751) |
| Sliding window | Sparse Transformer — one head attends to the previous l positions | 23 Apr 2019 | [link](https://arxiv.org/abs/1904.10509) |
| Linear attention | Efficient Attention: Attention with Linear Complexities (Shen et al.) | 4 Dec 2018 | [link](https://arxiv.org/abs/1812.01243) |
| Linear attention | Fast-weight memories (Schmidhuber, Neural Computation 1992) | 1992 | [link](https://doi.org/10.1162/neco.1992.4.1.131) |
| Delta rule | Widrow & Hoff, 'Adaptive switching circuits' (IRE WESCON) — the delta / LMS learning rule | 1960 | bibliographic record |
| RoPE | Jianlin Su's Chinese-language blog posts introducing rotary position embedding (kexue.fm / spaces.ac.cn) | 2021 | [link](https://spaces.ac.cn/archives/8265) |
| ALiBi | T5 — a learned scalar bias per head added to attention logits, indexed by bucketed relative distance | 23 Oct 2019 | [link](https://arxiv.org/abs/1910.10683) |
| FlashAttention | Online normalizer calculation for softmax (Milakov & Gimelshein) | 8 May 2018 | [link](https://arxiv.org/abs/1805.02867) |
| FlashAttention | Self-attention Does Not Need O(n²) Memory (Rabe & Staats) | 10 Dec 2021 | [link](https://arxiv.org/abs/2112.05682) |
| Hybrid schedules | GPT-3 alternates dense and locally banded attention layers | 28 May 2020 | [link](https://arxiv.org/abs/2005.14165) |
| Hybrid schedules | SPADE — a state-space layer at the bottom of an otherwise (local-)attention Transformer | 15 Dec 2022 | [link](https://arxiv.org/abs/2212.08136) |
| GQA | Multi-query attention (Shazeer) | 6 Nov 2019 | [link](https://arxiv.org/abs/1911.02150) |
| Position Interpolation | kaiokendev, 'Things I'm Learning While Training SuperHOT' — linear scaling of RoPE positions | 20 Jun 2023 | [link](https://kaiokendev.github.io/context) |
| NTK-aware | Position Interpolation / SuperHOT linear scaling | 27 Jun 2023 | [link](https://arxiv.org/abs/2306.15595) |
| YaRN | bloc97's NTK-by-parts proposal (GitHub PR, cited by YaRN) | 2023 | [link](https://arxiv.org/abs/2309.00071) |
| Attention sinks | What Does BERT Look At? — heads attend heavily to delimiter tokens | 11 Jun 2019 | [link](https://arxiv.org/abs/1906.04341) |
| Attention sinks | Quantizable Transformers — heads that 'do nothing' create outliers | 22 Jun 2023 | [link](https://arxiv.org/abs/2306.12929) |
| Attention sinks | Evan Miller, 'Attention Is Off By One' (blog) | 24 Jul 2023 | [link](https://www.evanmiller.org/attention-is-off-by-one.html) |
| Attention sinks | Vision Transformers Need Registers | 28 Sep 2023 | [link](https://arxiv.org/abs/2309.16588) |
| Mamba | RetNet — a fixed (not input-dependent) decay on a linear-attention state | 17 Jul 2023 | [link](https://arxiv.org/abs/2307.08621) |
| Mamba | Random Feature Attention — optional gate for recency | 3 Mar 2021 | [link](https://arxiv.org/abs/2103.02143) |
| Gated DeltaNet | Parallelizing Linear Transformers with the Delta Rule (Yang et al.) | 10 Jun 2024 | [link](https://arxiv.org/abs/2406.06484) |
| Gated DeltaNet | Mamba-2 | 31 May 2024 | [link](https://arxiv.org/abs/2405.21060) |
| NSA | Explicit Sparse Transformer (top-k attention) | 25 Dec 2019 | [link](https://arxiv.org/abs/1912.11637) |
| NSA | Liu et al. memory-compressed attention (strided convolution over K/V) | 30 Jan 2018 | [link](https://arxiv.org/abs/1801.10198) |
| DSA | Native Sparse Attention | 16 Feb 2025 | [link](https://arxiv.org/abs/2502.11089) |
| DroPE | NoPE (Haviv et al.; Kazemnejad et al.) | 30 Mar 2022 | [link](https://arxiv.org/abs/2203.16634) |
| DeepSeek-V4 CSA/HCA | DeepSeek Sparse Attention (V3.2-Exp) | 29 Sep 2025 | [link](https://api-docs.deepseek.com/news/news250929) |
| DeepSeek-V4 CSA/HCA | NSA's compressed-token branch | 16 Feb 2025 | [link](https://arxiv.org/abs/2502.11089) |

### First notable use (used for the "ideas wait for their bill" analysis)

| Mechanism | First notable use | Date | Source |
|---|---|---|---|
| Learned positions | GPT-1 | 11 Jun 2018 | [link](https://openai.com/index/language-unsupervised/) |
| Sparse Transformer | GPT-3 (alternating dense / banded layers) | 28 May 2020 | [link](https://arxiv.org/abs/2005.14165) |
| MQA | PaLM | 5 Apr 2022 | [link](https://arxiv.org/abs/2204.02311) |
| Top-k attention | DeepSeek-V3.2-Exp (indexer + top-k) | 29 Sep 2025 | [link](https://api-docs.deepseek.com/news/news250929) |
| Sliding window | Mistral 7B | 10 Oct 2023 | [link](https://arxiv.org/abs/2310.06825) |
| Linear attention | MiniMax-01 (lightning attention hybrid) | 14 Jan 2025 | [link](https://arxiv.org/abs/2501.08313) |
| Delta rule | Qwen3-Next (Gated DeltaNet layers) | 11 Sep 2025 | [link](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |
| RoPE | GPT-NeoX / Mesh Transformer JAX (EleutherAI) | 20 Apr 2021 | [link](https://blog.eleuther.ai/rotary-embeddings/) |
| ALiBi | BLOOM | Jul 2022 | [link](https://huggingface.co/bigscience/bloom) |
| Hybrid schedules | Jamba | 28 Mar 2024 | [link](https://arxiv.org/abs/2403.19887) |
| GQA | Llama 2 (34B, 70B) | 18 Jul 2023 | [link](https://arxiv.org/abs/2307.09288) |
| NTK-aware | Code Llama (raised RoPE base) | 24 Aug 2023 | [link](https://arxiv.org/abs/2308.12950) |
| YaRN | DeepSeek-V2 | 7 May 2024 | [link](https://arxiv.org/abs/2405.04434) |
| Attention sinks | gpt-oss (learned sink bias, model card) | 8 Aug 2025 | [link](https://arxiv.org/abs/2508.10925) |
| Mamba | Jamba | 28 Mar 2024 | [link](https://arxiv.org/abs/2403.19887) |
| MLA | DeepSeek-V2 | 7 May 2024 | [link](https://arxiv.org/abs/2405.04434) |
| Gated DeltaNet | Qwen3-Next | 11 Sep 2025 | [link](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) |
| DSA | DeepSeek-V3.2-Exp | 29 Sep 2025 | [link](https://api-docs.deepseek.com/news/news250929) |
| DeepSeek-V4 CSA/HCA | DeepSeek-V4 Preview | 24 Apr 2026 | [link](https://api-docs.deepseek.com/news/news260424) |
<!-- sources:end -->

## Mechanisms added for the story

I started from a core list of mechanisms and covered it in full. These were added because a later mechanism makes no sense without them (each has a card, a date and a primary source above):

- **Additive attention** (Bahdanau et al., arXiv:1409.0473): the prologue. Attention was invented to escape a fixed-size memory, which linear attention later re-accepts on purpose.
- **Relative position representations** (Shaw et al., arXiv:1803.02155) and **NoPE** (Haviv et al., arXiv:2203.16634): the position thread that RoPE, ALiBi and DroPE continue.
- **Transformer-XL** (arXiv:1901.02860): the first widely used answer to "what crosses the window boundary".
- **FlashAttention** (Dao et al., arXiv:2205.14135, v1 27 May 2022): the most consequential omission. It makes exact attention cheap in memory traffic, which changed which bill mattered.
- **Hybrid stacks** (H3, arXiv:2212.14052), **Position Interpolation** (arXiv:2306.15595) and **Mamba** (arXiv:2312.00752): the roots of mixed-depth layer schedules, of NTK-aware scaling and YaRN, and of Gated DeltaNet.
- "Compressed and sparse attention as DeepSeek does it" is shown as three dated stops: **NSA** (Feb 2025), **DSA** (Sep 2025) and **CSA/HCA in DeepSeek-V4** (Apr 2026).

## Checks on the notes I learned from

Details are on the page under "Checking the notes I learned from".

- **Refined:** the notes say the available record does not establish the DroPE algorithm. The method is published (Gelberg et al., arXiv:2512.12167, 13 Dec 2025): remove RoPE from every layer after pretraining, then recalibrate briefly at the original context length. That is consistent with the notes' caution that it is a training step, not an inference switch.
- **Refined:** "attention does not know order" is exactly right for the scores and for bidirectional encoders, but a causal decoder's mask leaks position (Haviv et al., 2022). This is why DroPE can work.
- **Confirmed:** MQA predates GQA by three and a half years (teaching it as GQA's extreme is still the right teaching order); DeepSeek-V4 applies RoPE to the last 64 dimensions and uses compressed blocks plus a top-k indexer interleaved with a heavily compressed dense form; the KV-cache yardstick (6.44 GB per user, 51.54 GB for eight) is reproduced exactly by the site's formula and its tests.

## Technical choices

- **Vite + TypeScript, no UI framework.** Each figure is a function that builds its own DOM/SVG (or canvas, for the attention engine) and re-renders on input. About 70 KB of gzipped JavaScript for the whole site. Figures mount lazily as they approach the viewport.
- **Real arithmetic.** `src/lib/math.ts` holds the maths every figure uses (softmax, RoPE and its PI/NTK/YaRN scalings, ALiBi slopes, linear-attention state, delta-rule writes, the KV-cache formula). `tests/math.test.ts` checks the identities the page claims, including the worked examples (140, 95 vs 55, 6.44 GB).
- **Accessibility and motion.** Controls are native inputs and buttons with labels. Animations respect `prefers-reduced-motion` (they jump to the end state). Light and dark themes. Wide figures pan on phones instead of shrinking their labels.
- **QA.** `scripts/qa.mjs` drives a local Chrome at four viewport sizes and both colour schemes, checking console errors, horizontal overflow, date order, that every figure mounted, broken anchors and illegibly small labels, and saving screenshots.

## Repository layout

```
src/data/ledger.json         dates, sources, earlier roots, first uses (single source of truth)
src/content/entries.ts       narrative for each mechanism (no hard-coded dates)
src/content/eras.ts          era interludes
src/widgets/*.ts             interactive figures (engine.ts is the attention engine)
src/lib/math.ts              the maths, tested in tests/math.test.ts
src/main.ts, src/sections.ts page assembly, map, timeline rail, synthesis, sources
scripts/check-ledger.mjs     build-time date/text consistency audit
scripts/gen-sources.mjs      regenerates the README source tables from the ledger
scripts/qa.mjs, snap.mjs     browser QA and screenshots
```

## Credits

Built for ERA V5 Session 8 (The School of AI). Interactive figures use small hand-made toy numbers unless a caption says otherwise. Formulas and reported figures come from the cited papers.
