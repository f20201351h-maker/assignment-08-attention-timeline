// Narrative for each timeline entry. Dates never appear here as literals for
// entries: they come from src/data/ledger.json. scripts/check-ledger.mjs fails
// the build if any year written in this file is unknown to the ledger.

import { attentionWidget } from '../widgets/attention.ts';
import { patternWidget } from '../widgets/patterns.ts';
import { headsWidget, mlaWidget } from '../widgets/heads.ts';
import { topkWidget } from '../widgets/topk.ts';
import { linearWidget } from '../widgets/linear.ts';
import { deltaWidget } from '../widgets/delta.ts';
import { learnedWallWidget, sinusoidWidget, ropeWidget, alibiWidget, nopeWidget } from '../widgets/positions.ts';
import { extensionWidget } from '../widgets/extension.ts';
import { sinkWidget } from '../widgets/sinks.ts';
import { flashWidget } from '../widgets/flash.ts';
import { scheduleWidget } from '../widgets/schedule.ts';
import { deepseekWidget } from '../widgets/deepseek.ts';
import { dropeWidget, selectiveWidget, bottleneckWidget } from '../widgets/misc.ts';

export interface Story {
  oneliner: string;
  moment: string;          // what existed, what hurt
  change: string;          // what changed mechanically
  widget?: () => HTMLElement;
  buys: string[];
  costs: string[];
  choose: string;          // when would I actually pick it
  next: { id?: string; text: string };
}

export const STORIES: Record<string, Story> = {
  additive: {
    oneliner: 'A translator that is allowed to look back.',
    moment: `<p>Neural machine translation in the mid-2010s was an encoder–decoder pair of recurrent networks. The encoder read the whole source sentence and squeezed it into <em>one fixed-length vector</em>; the decoder wrote the translation from that vector alone. Bahdanau, Cho and Bengio name the problem directly: that vector is a bottleneck. Long sentences simply do not fit.</p>`,
    change: `<p>Keep every encoder state instead of only the last. At each output step, score every source state against what the decoder currently needs, softmax the scores, and take the weighted sum as a fresh “context” for that step. Their scorer is a small feed-forward network (vᵀ tanh(W s + U h)) — hence <strong>additive</strong> attention. Nobody tells the model which words align; it learns soft alignments as a side effect.</p>
    <p>The shape you will see in every entry below is already here: <strong>score → softmax → weighted sum</strong>.</p>`,
    widget: bottleneckWidget,
    buys: ['No single fixed-size summary: each output word gets its own view of the source.', 'Big gains on long sentences.', 'Interpretable alignments for free.'],
    costs: ['Every output step reads every source state: work grows with source length × target length.', 'All encoder states must be kept around.', 'Still a recurrent network — training is sequential over time, so it does not use parallel hardware well.'],
    choose: 'Today, as history: you would not build this. But keep its trade in mind — it gave up a fixed-size memory to get exact access. Six years later linear attention (and soon the delta rule) takes the opposite side of that same trade, on purpose.',
    next: { id: 'sdpa', text: 'Three years later the Transformer removes the recurrent network entirely and makes attention the whole model.' },
  },

  'learned-abs': {
    oneliner: 'Give every slot its own learned vector.',
    moment: `<p>The moment a model processes all positions in parallel — a convolutional encoder, and soon the Transformer — it loses the one thing recurrence gave for free: order. Gehring et al. note that averaging nearby word embeddings “does not convey positional information”. Attention by itself is no better: shuffle the tokens and every score shuffles with them.</p>`,
    change: `<p>Keep a second table with one learned row per position, and add row <code>t</code> to the token embedding at position <code>t</code>. That is all. The ConvS2S paper the Transformer cites does the same, the Transformer authors tried it and found it about as good as their sinusoids, and GPT-1, BERT and GPT-2 used it.</p>`,
    widget: learnedWallWidget,
    buys: ['As simple as it gets: one gather and one add.', 'Maximally flexible inside the trained range — any per-position quirk can be learned.', 'Negligible parameters next to the vocabulary table.'],
    costs: ['A hard wall: no row exists past the table size, and rows past the training length were never trained (the “wall at max_position”).', 'Purely absolute: “three tokens back” has to be learned separately at every absolute position.', 'Position and meaning share one vector, which every layer has to keep disentangling.'],
    choose: 'Fixed-length encoders and classifiers whose inputs never exceed the trained length (a 512-token BERT-style model), small experiments. Not for anything expected to run longer than it trained.',
    next: { id: 'sinusoidal', text: 'The Transformer paper proposes computing position instead of storing it.' },
  },

  sdpa: {
    oneliner: 'Drop the recurrent network. Attention becomes the whole layer.',
    moment: `<p>Sequence models were recurrent networks with attention bolted on (slow, sequential training; long paths between distant tokens) or convolutions (parallel, but each layer only sees a local patch). Vaswani et al. wanted both parallel training and a short path between any two tokens.</p>`,
    change: `<p>Every token projects itself into a query, a key and a value. Each query is compared with every key by a dot product, divided by √dₖ, masked so a decoder cannot read the future, softmaxed into weights that sum to one, and used to average the values. <strong>Multi-head</strong> attention runs several smaller copies side by side with their own projections and concatenates the results. Nothing is recurrent, so a whole training sequence is processed at once. The foundation section at the top of this page walks through every number.</p>`,
    buys: ['One hop between any two tokens, however far apart.', 'Fully parallel training — a perfect fit for GPUs.', 'Exact, query-specific selection: each query forms a fresh distribution over every earlier token.'],
    costs: ['Compute: every query meets every key — T² scores per head per layer.', 'Memory: generation needs every earlier key and value kept around (the KV cache), growing with T for every active user.', 'No sense of order: the scores are content-only unless position is supplied.'],
    choose: 'The default whenever T² and a full cache are affordable: short and medium contexts, most training runs, and anywhere exact recall of a specific earlier token matters. Every entry after this one is somebody looking at its bill and trying to pay less of it.',
    next: { text: 'The bill arrives in two parts — compute and cache — and the field attacks them separately. Position comes first, because the model cannot work without it.' },
  },

  sinusoidal: {
    oneliner: 'Compute position instead of storing it.',
    moment: `<p>The same paper had to tell an order-blind layer where each token is. A learned table works but has a last row.</p>`,
    change: `<p>Add a fixed signal: dimension pair <code>i</code> holds sin and cos of <code>pos / 10000^(2i/d)</code>. Think of a bank of clock hands turning at geometrically spaced speeds; a position is a reading of all the hands. Because PE(pos + k) is a fixed linear function of PE(pos), the authors hoped relative offsets would be easy to learn, and chose sinusoids because they “may” let the model extrapolate to longer sequences.</p>`,
    widget: sinusoidWidget,
    buys: ['Zero parameters.', 'Defined for every position — no table to run off the end of.', 'Smooth multi-scale structure: fast hands resolve neighbours, slow hands span long ranges.'],
    costs: ['Still absolute, and still added into the content vector.', '“Defined everywhere” is not “works everywhere”: extrapolation was a hypothesis, and later measurements (the ALiBi paper) found it does not hold up well past the training length.', 'In the original experiments it was no better than a learned table — the choice was about hope, not measured quality.'],
    choose: 'Mostly historical now; a reasonable zero-parameter baseline for fixed-length encoders.',
    next: { id: 'relative', text: 'The next step stops labelling absolute slots and encodes distance inside attention.' },
  },

  relative: {
    oneliner: 'What matters is how far apart, not where.',
    moment: `<p>With absolute signals the model has to learn “the word three back” as a different pattern at every absolute position. Language mostly cares about relations.</p>`,
    change: `<p>Shaw, Uszkoreit and Vaswani add a learned vector for each (clipped) relative distance directly into the attention computation for each query–key pair, so the score for tokens <code>i</code> and <code>j</code> depends on their content and on <code>i − j</code>. Distances beyond a cut-off share one vector.</p>`,
    buys: ['Translation-invariant relations: the same pattern works at any offset.', 'Better translation quality than absolute positions in their experiments.', 'Nothing tied to a maximum absolute length.'],
    costs: ['Extra per-pair terms inside attention: more memory traffic and harder to fuse into fast kernels.', 'Clipping: every distance past the cut-off looks the same.', 'Still learned parameters per distance bucket.'],
    choose: 'Encoder models of its era. Its descendants (T5’s learned per-distance bias, then ALiBi and RoPE) are what you would actually use.',
    next: { id: 'rope', text: 'RoPE later gets relative position out of the dot product itself, with no extra terms at all.' },
  },

  'transformer-xl': {
    oneliner: 'Let the next window see the last one.',
    moment: `<p>Language models trained on fixed-length segments start every segment cold: nothing crosses the boundary. Long documents get chopped into pieces that cannot see each other.</p>`,
    change: `<p>Cache the hidden states of the previous segment and let the current segment attend to them, without back-propagating into them (stop-gradient). Because reused states would collide with absolute positions, Transformer-XL also switches to a relative position scheme. Each layer can reach one segment further back, so reach grows with depth.</p>`,
    buys: ['Information crosses segment boundaries.', 'Reach longer than the training window at modest extra cost.', 'Faster evaluation: earlier states are reused instead of recomputed.'],
    costs: ['Extra memory for the cached states.', 'Gradients never flow into the past, so long dependencies are learned only indirectly.', 'Reach is still bounded, and position handling becomes more complex.'],
    choose: 'Streaming language modelling over long text with bounded memory, when approximate memory of the previous segment is enough. The instinct reappears in memory-stream designs: one summary vector carried across a chunk boundary, written with stop-gradient.',
    next: { id: 'sparse-transformer', text: 'Meanwhile, the T² bill itself becomes the target.' },
  },

  'sparse-transformer': {
    oneliner: 'Compute only a structured subset of the pairs.',
    moment: `<p>OpenAI wanted to generate very long sequences — images as pixels, raw audio, long text — with tens of thousands of positions. A full T × T score matrix per layer and head was out of reach.</p>`,
    change: `<p>Factorize attention across heads. In the <em>strided</em> pattern one head looks at the previous <code>l</code> positions and another at every <code>l</code>-th position; with <code>l ≈ √T</code>, any token reaches any earlier token in two hops. The <em>fixed</em> pattern uses blocks plus summary positions. Total work drops from T² to about T√T.</p>`,
    widget: () => patternWidget('strided'),
    buys: ['Much longer sequences on the same hardware.', 'Exact softmax over whatever is selected.', 'Every token still connected to every other within two hops.'],
    costs: ['The pattern is decided by position, not content: the one key that matters may only be reachable indirectly.', 'Needs custom kernels.', 'Works best when the data’s structure matches the pattern (rows of an image, beats of music).'],
    choose: 'Very long, regularly structured data, or as some of the layers in a model: GPT-3 alternated dense and locally banded layers.',
    next: { text: 'Three other answers follow within a year: share keys and values (MQA), select keys by content (top-k), and drop softmax altogether (linear attention).' },
  },

  mqa: {
    oneliner: 'All query heads share one set of keys and values.',
    moment: `<p>Training was now fast and parallel, but generation is one token at a time. Shazeer measured where incremental decoding spends its time: reloading the large key and value tensors of every head from memory at every step. The bottleneck was memory bandwidth, not arithmetic.</p>`,
    change: `<p>Keep all the query heads — different questions are cheap — but give the layer a single key head and a single value head shared by every query head. The cache and the bytes moved per step shrink by the number of heads.</p>`,
    widget: () => headsWidget('1'),
    buys: ['Cache and decoding bandwidth divided by the head count.', 'Much faster incremental decoding, larger batches per accelerator.', 'Tiny change to the architecture.'],
    costs: ['Every head searches the same keys and values: less diversity. The GQA authors later report quality degradation and training instability.', 'Converting an existing multi-head model needs retraining.', 'The cache still grows linearly with context.'],
    choose: 'Latency-critical serving where cache and bandwidth dominate and a small quality loss is acceptable — PaLM used it. For most new dense models today, GQA is the better-balanced version of the same idea.',
    next: { id: 'gqa', text: 'Most models ignored this for years — until serving cost became the main cost.' },
  },

  topk: {
    oneliner: 'Let each query keep only its k best matches.',
    moment: `<p>Softmax never gives exactly zero: every query spreads some weight over every token, relevant or not. Zhao et al. framed this as attention not being concentrated enough — a quality problem, not a speed problem.</p>`,
    change: `<p>Compute all the scores, keep the <code>k</code> largest per query, set the rest to −∞, then softmax. Only <code>k</code> values are mixed.</p>`,
    widget: topkWidget,
    buys: ['Sharper attention with less noise from irrelevant tokens.', 'Fewer values mixed per query.', 'The seed of content-based sparsity: which keys are read is decided by what they contain.'],
    costs: ['The catch: to know the top k you first scored all T keys, so the expensive part is untouched.', 'Dropped keys receive no gradient, so the model cannot easily learn that it should have kept them.', 'Queries that need evidence from many places lose information; selection itself is awkward on GPUs.'],
    choose: 'On its own, rarely. It becomes worth it once something cheaper than full scoring proposes the candidates — LSH buckets and clustering tried in 2020, trained indexers made it work in 2025.',
    next: { id: 'nsa', text: 'Five years later DeepSeek gives top-k the cheap selector it always lacked.' },
  },

  'sliding-window': {
    oneliner: 'Each token reads only its recent neighbours.',
    moment: `<p>BERT-style models were capped at 512 tokens; documents are much longer. Full attention over thousands of tokens was too expensive, and a lot of language is local anyway.</p>`,
    change: `<p>A band mask: token <code>i</code> attends only to the previous <code>W</code> tokens. Longformer adds dilated windows and a few task-specific global tokens. Stacking layers lets information hop: after <code>L</code> layers a token can be influenced by roughly <code>L × W</code> positions back. For generation, the cache can become a rolling buffer of the last <code>W</code> entries (Mistral 7B).</p>`,
    widget: () => patternWidget('window'),
    buys: ['Work O(T · W) instead of T².', 'Cache capped at W per layer during generation.', 'Simple to implement efficiently.'],
    costs: ['No direct access beyond the window: distant facts must hop through layers and get diluted.', 'Exact retrieval of a far-back token suffers — the “L × W” reach is an upper bound, not a guarantee.', 'Pure windows break in streaming once the first tokens fall out (see attention sinks).'],
    choose: 'When dependencies are mostly local and memory must be bounded. Today it is most often interleaved with global layers (Gemma 2 alternates local and global layers) rather than used alone.',
    next: { id: 'sinks', text: 'Three years later, streaming with a pure window reveals a strange failure.' },
  },

  linear: {
    oneliner: 'Remove softmax, regroup, and attention becomes a recurrent network.',
    moment: `<p>The “efficient transformer” rush: many groups attacking T² at once. The key experiment can be done by hand — with softmax removed, the query can be factored out.</p>`,
    change: `<p>Replace exp(q·k) with a product of feature maps φ(q)·φ(k) (Katharopoulos et al. use elu + 1, keeping it positive) and normalise. Then Σⱼ φ(q)·φ(kⱼ) vⱼ = (Σⱼ vⱼ φ(kⱼ)ᵀ) φ(q): the bracket is a running state <code>S</code> of fixed size, updated once per token. Training is linear in T; generation costs the same for token one and token one million.</p>`,
    widget: linearWidget,
    buys: ['Linear-time training, constant-memory generation: no growing cache.', 'Huge speedups for very long autoregressive generation.', 'A clean bridge between Transformers and recurrent networks.'],
    costs: ['It is not softmax: weights are flatter and there is no competition between keys.', 'Every memory is superposed in one fixed matrix, so associations interfere and capacity is limited.', 'Weak at exact recall and copying; on language modelling, follow-up work (e.g. Schlag et al.) measured a quality gap to softmax.'],
    choose: 'Very long sequences where a summary is enough and exact token recall is not the point. In 2025-era production models it appears as most of the layers of a hybrid, rarely alone.',
    next: { id: 'delta', text: 'The next question is how to write into that fixed state without making a mess.' },
  },

  delta: {
    oneliner: 'Read the memory first; write only the error.',
    moment: `<p>Schlag, Irie and Schmidhuber show linear attention is a “fast weight programmer” whose memory is written by pure addition. Additive writes can never correct an old association, and once there are more stored pairs than dimensions, writes overwrite each other blindly.</p>`,
    change: `<p>Before writing, read what the memory currently returns for the key, <code>v_old = S k</code>. Then write only a fraction β of the difference: <code>S ← S + β (v − v_old) kᵀ</code>, with β produced by the model. It is the delta rule of adaptive filters, used here as the write rule of an attention memory. A worked example: memory says 40, the answer should be 55, write +15 rather than +55.</p>`,
    widget: () => deltaWidget('delta'),
    buys: ['Old associations can be corrected instead of piling up.', 'Better use of a fixed capacity.', 'Still constant memory per token.'],
    costs: ['The update depends on the current state, so it is no longer a simple running sum — the 2021 form had no efficient parallel training algorithm.', 'Still a fixed, lossy memory.', 'No way to clear many stale associations at once (no global forgetting).'],
    choose: 'Not on its own today; it is the core of Gated DeltaNet layers in current hybrids.',
    next: { id: 'gated-deltanet', text: 'It waits three years for a fast training algorithm and a forgetting gate.' },
  },

  rope: {
    oneliner: 'Encode position as a rotation, so dot products see only distance.',
    moment: `<p>Absolute schemes mix position into content. Relative schemes add per-pair terms that complicate fast kernels and caching. The ideal is relative position with no extra terms at all.</p>`,
    change: `<p>Split each query and key into 2-d pairs. Rotate pair <code>i</code> of a token at position <code>m</code> by angle <code>m·θᵢ</code>, with θᵢ geometrically spaced like the sinusoid frequencies. Because a dot product depends only on the angle between two arrows, R(mθ)q · R(nθ)k = q · R((n − m)θ)k: the absolute rotations cancel and only the gap survives. Applied to queries and keys in every layer; values are untouched.</p>`,
    widget: ropeWidget,
    buys: ['Relative position for free, inside the ordinary dot product.', 'No parameters; compatible with fast kernels and with caching (a cached key is rotated once).', 'Many frequencies: fast pairs for local order, slow pairs for long range.'],
    costs: ['Computable at any position, but not reliable beyond the training length: slow pairs reach angles never seen in training. A whole subfield of extension methods exists because of this.', 'Interacts awkwardly with KV compression — MLA has to carry a separate uncompressed rotary key.', 'Partial-RoPE and base choices are extra design knobs.'],
    choose: 'The default for most open decoder LLMs. Pair it with an extension method (YaRN, or DroPE-style recalibration) if you need longer contexts than you trained on.',
    next: { id: 'alibi', text: 'Four months later, ALiBi argues for extrapolation directly — with no rotation at all.' },
  },

  alibi: {
    oneliner: 'No position vectors — just penalise distance.',
    moment: `<p>Training at long lengths is expensive. Press, Smith and Lewis ask whether a model trained short can be run long, and measure that sinusoidal, rotary and T5-style schemes do not extrapolate efficiently.</p>`,
    change: `<p>Add nothing to the embeddings. Instead subtract <code>m · (i − j)</code> from each attention score before softmax, with a fixed slope <code>m</code> per head (1/2, 1/4, … 1/256 for eight heads). Steep heads look locally, gentle heads can still look far.</p>`,
    widget: alibiWidget,
    buys: ['Perplexity stays stable on inputs longer than training: train short, test long.', 'Saves training compute by training at shorter lengths.', 'Essentially free to compute.'],
    costs: ['A built-in recency bias: far tokens are always penalised.', 'Stable perplexity on longer inputs is not the same as using information from far away — a model can “extrapolate” by largely ignoring distant tokens.', 'Did not become the default for the largest later models, which mostly settled on RoPE.'],
    choose: 'When inputs longer than training are expected and recency is the right prior. BLOOM and MPT used it.',
    next: { id: 'pi', text: 'The field mostly chose RoPE instead — and two years later had to learn how to stretch it.' },
  },

  nope: {
    oneliner: 'What if a causal model gets no position signal at all?',
    moment: `<p>Every scheme so far assumes attention is blind to order. For a causal decoder that is not quite true: the mask means token <code>t</code> can see exactly <code>t</code> tokens.</p>`,
    change: `<p>Remove positional encodings entirely. Haviv et al. find causal language models without them remain competitive and still learn absolute position internally. Kazemnejad et al. later call this NoPE and find it length-generalises better than RoPE, ALiBi or absolute embeddings in small decoder-only experiments.</p>`,
    widget: nopeWidget,
    buys: ['No positional machinery at all — nothing to extrapolate.', 'Better length generalisation in small-scale studies.', 'A conceptual correction: for decoders, order is not entirely missing.'],
    costs: ['Only works for causal (masked) models; a bidirectional encoder really is order-blind.', 'Slightly behind explicit encodings in perplexity in the original study.', 'The model must spend capacity inferring what it was not told.'],
    choose: 'Research and ablations — and as the foundation of DroPE.',
    next: { id: 'drope', text: 'DroPE later uses RoPE to train and then deletes it, betting on exactly this property.' },
  },

  flash: {
    oneliner: 'Same exact attention, computed in tiles that never leave fast memory.',
    moment: `<p>By 2022 the efficient-attention papers had a problem: many approximations cut FLOPs on paper but rarely sped up real training. Dao et al. identify why — attention on GPUs was limited by reading and writing the T × T matrix to slow main memory, not by arithmetic.</p>`,
    change: `<p>Tile the computation: load blocks of queries, keys and values into fast on-chip memory, compute scores, and fold them into a running max and running sum per row (the online-softmax trick), rescaling earlier partial sums when a larger score appears. The T × T matrix is never written out; the backward pass recomputes it tile by tile. The result is mathematically the same attention — equal up to floating-point rounding, not an approximation.</p>`,
    widget: flashWidget,
    buys: ['Exact — no approximation at all.', 'Activation memory linear in T instead of quadratic; real wall-clock speedups.', 'Made longer training contexts practical, and shifted the field’s sense of what is actually expensive.'],
    costs: ['FLOPs are still quadratic.', 'The inference KV cache is exactly as large as before.', 'Hardware-specific kernel engineering; architectures that do not fit the kernel pay a real price, which quietly pushes research towards kernel-friendly designs.'],
    choose: 'Always, if your hardware supports it — it is an implementation of the same mathematics, not a modelling choice.',
    next: { id: 'gqa', text: 'With training compute eased, the binding cost moves to serving: the cache.' },
  },

  hybrid: {
    oneliner: 'Don’t choose one memory system — stack them.',
    moment: `<p>State-space and linear layers were cheap but poor at recalling and comparing specific earlier tokens; the H3 authors single out exactly these synthetic recall tasks. Attention is excellent at them and expensive.</p>`,
    change: `<p>Use cheap fixed-state layers for most of the depth and keep a few softmax-attention layers for exact lookups. H3’s hybrid keeps just two attention layers. Later production models fix a repeating ratio: Jamba 1 attention to 7 Mamba layers, Qwen3-Next and Kimi Linear 3 state layers to 1 attention layer — the same 3 : 1 shape as a DDDGDDDG layer schedule.</p>`,
    widget: scheduleWidget,
    buys: ['Most layers have constant memory; only the few attention layers keep a growing cache.', 'The attention layers restore exact recall that pure state models lack.', 'Reported quality matching or beating pure Transformers at lower serving cost.'],
    costs: ['The attention layers’ cache still grows with context.', 'Ratio and placement are expensive-to-ablate hyperparameters (Jamba saw little difference between 1:3 and 1:7; Kimi Linear ablated 0:1 up to 15:1 and found 3:1 best, with 7:1 clearly worse on validation).', 'Two kinds of kernels and state to maintain.'],
    choose: 'Long-context serving at scale where memory dominates and you can afford ablations to pick the ratio.',
    next: { id: 'gated-deltanet', text: 'Which state layer to use is settled later: Gated DeltaNet.' },
  },

  gqa: {
    oneliner: 'Share keys and values within groups of query heads.',
    moment: `<p>Large models were now served to millions of users, and the per-user KV cache — not training — was the cost that hurt. MQA saved the most but could hurt quality, and nobody wanted to retrain a flagship from scratch for it.</p>`,
    change: `<p>Split the query heads into <code>G</code> groups; each group shares one key head and one value head (G = H is ordinary multi-head, G = 1 is MQA). Existing multi-head checkpoints can be “uptrained”: mean-pool the key/value heads in each group and continue training with about 5% of the original pretraining compute.</p>`,
    widget: () => headsWidget('2'),
    buys: ['Close to multi-head quality at close to multi-query speed (the paper’s claim).', 'Cache divided by H / G.', 'Existing models can be converted cheaply.'],
    costs: ['A constant-factor saving only: the cache still grows linearly with context (it lowers the slope, it does not stop the line).', 'Some quality is still traded against MHA.', 'One more knob (G) to choose.'],
    choose: 'The default for dense Transformers whenever serving cost matters, from small models to the largest. It is the baseline rather than the answer.',
    next: { id: 'mla', text: 'MLA asks for much more compression without the quality loss.' },
  },

  pi: {
    oneliner: 'Squeeze the new positions into the old range.',
    moment: `<p>Open models trained with RoPE at a few thousand tokens were suddenly everywhere, and everyone wanted longer contexts without retraining. Running RoPE past the trained length fails badly: slow rotary pairs reach angles the model has never seen.</p>`,
    change: `<p>Divide every position index by the extension factor <code>s</code>, so the longest new context maps onto angles inside the trained range, then fine-tune briefly. A hobbyist (kaiokendev) published the same linear scaling a week earlier.</p>`,
    widget: () => extensionWidget('pi'),
    buys: ['Context extended several-fold with a short fine-tune.', 'No new parameters or architecture changes.', 'No unseen angles anywhere.'],
    costs: ['Every frequency is squeezed, including the fast ones that separate neighbouring tokens: local resolution drops.', 'Needs some fine-tuning to recover quality.', 'Slight degradation inside the original window.'],
    choose: 'Quick, simple extension when you can afford a short fine-tune.',
    next: { id: 'ntk', text: 'Two days later, a Reddit post argues that the fast frequencies should be left alone.' },
  },

  ntk: {
    oneliner: 'Change RoPE’s base, not its positions.',
    moment: `<p>Position Interpolation crushes high frequencies. The local-model community wanted extension without fine-tuning.</p>`,
    change: `<p>Raise the RoPE base: <code>b′ = b · s^(d/(d−2))</code>. The fastest pairs barely change (local detail preserved) while the slowest are interpolated about as much as PI would. A day later a “dynamic” version adjusted the scale to the current length.</p>`,
    widget: () => extensionWidget('ntk'),
    buys: ['Reasonable extension with no fine-tuning at modest factors.', 'Keeps local resolution.', 'One-line change.'],
    costs: ['A heuristic: some middle frequencies still drift slightly out of their trained range (YaRN’s critique).', 'Falls behind fine-tuned methods at large factors.', 'Its primary source is a forum post, not a paper — which is itself part of the history.'],
    choose: 'Quick inference-time stretching by a small factor. Raising the base before long-context fine-tuning (as Code Llama did) is the trained cousin of this idea.',
    next: { id: 'yarn', text: 'YaRN turns these intuitions into a per-frequency recipe.' },
  },

  yarn: {
    oneliner: 'Treat each RoPE frequency according to how much of it training saw.',
    moment: `<p>PI and NTK-aware scaling were each half right: one protects the slow pairs, the other the fast ones.</p>`,
    change: `<p><strong>NTK-by-parts:</strong> pairs that completed many full rotations within the training length are left alone; pairs that never completed one are fully interpolated; a linear ramp in between. Plus an <strong>attention temperature</strong>: queries and keys are each scaled by √(1/t) = 0.1 ln s + 1 (so the logits grow by its square) as the context stretches, so attention does not flatten out. A dynamic variant adjusts the scale to the current length at inference.</p>`,
    widget: () => extensionWidget('yarn'),
    buys: ['Extension with much less fine-tuning data than earlier methods (per the paper).', 'Keeps both local resolution and long-range coverage.', 'Widely adopted: DeepSeek-V2 extends 4K → 128K with it; gpt-oss uses it for its dense layers.'],
    costs: ['Several per-model hyperparameters (ramp bounds, temperature).', 'Still stretching a model trained short; reach is evidence for a specific model, not a guarantee.', 'Does nothing for compute or cache: a 128K context still costs 128K tokens of KV.'],
    choose: 'The standard way to extend a RoPE model by roughly 4–32× when you can do a short fine-tune.',
    next: { id: 'drope', text: 'The extension story will eventually ask whether to stretch RoPE at all.' },
  },

  sinks: {
    oneliner: 'Keep the first few tokens, and a window can stream forever.',
    moment: `<p>Chat and streaming deployments want bounded memory. The obvious fix — keep only a sliding window of recent keys and values — collapses the moment the very first token is evicted.</p>`,
    change: `<p>Xiao et al. explain why: softmax must hand out exactly 1, and many heads park their spare attention on the first tokens, which every later token can see. Evict them and that mass is forced onto the window. Their fix is almost embarrassingly small: always keep the keys and values of the first 4 tokens (the “sinks”) plus a rolling window. Optionally, pre-train with a dedicated sink token.</p>`,
    widget: sinkWidget,
    buys: ['Constant memory with stable language modelling over millions of streamed tokens.', 'No fine-tuning needed.', 'Explains a widely seen phenomenon (attention piling onto the first token).'],
    costs: ['It does <em>not</em> extend memory: tokens evicted from the middle are gone for good.', 'Stable perplexity is not long-context understanding.', 'It works around a quirk of softmax rather than removing it; later designs remove the sink with an output gate or a learned extra term in the softmax denominator.'],
    choose: 'Endless streaming (assistants, logs, live transcription) where only recent context matters. Not for questions about something said 50,000 tokens ago.',
    next: { id: 'mla', text: 'Bounded memory by eviction forgets. The next entries try to shrink memory without forgetting.' },
  },

  mamba: {
    oneliner: 'A recurrent state that decides, token by token, what to keep.',
    moment: `<p>Linear attention and earlier state-space models update their state with the same dynamics for every token. They cannot choose to ignore filler and hold on to a fact, and on language they lagged behind Transformers.</p>`,
    change: `<p>Make the state-space parameters functions of the current input, so how much to write and how fast to forget depends on the token itself (“selective”). A hardware-aware parallel scan keeps training fast. There is no attention in the model at all.</p>`,
    widget: selectiveWidget,
    buys: ['Linear-time training, constant memory per token at inference.', 'Content-dependent memory: can keep what matters across long stretches of noise.', 'Competitive language modelling at the scales it was tested.'],
    costs: ['A fixed-size state still cannot hold everything: exact recall and copying from long contexts is the weak spot.', 'Not attention — retrieving one specific earlier token is lossy by design.', 'New kernels and a less mature ecosystem.'],
    choose: 'Long streams where a learned summary suffices (audio, genomics, long logs) — or as the bulk of a hybrid.',
    next: { id: 'gated-deltanet', text: 'Mamba-2 then shows these state models and linear attention are two views of the same thing, and Gated DeltaNet merges Mamba-2’s gate with the delta rule.' },
  },

  mla: {
    oneliner: 'Cache a small latent; rebuild keys and values from it.',
    moment: `<p>GQA and MQA cut the cache by sharing heads, at some cost in quality. DeepSeek wanted multi-head quality with a far smaller cache for a very large model.</p>`,
    change: `<p>Project each token’s hidden state down to a small latent vector <code>c_KV</code> (512 numbers in DeepSeek-V2) and cache only that. Per-head keys and values are re-expanded from it; at inference the up-projection matrices can be folded into the query and output projections, so the full keys and values never need to be materialised. RoPE breaks that folding (a position-dependent rotation would sit between the matrices), so position travels in a separate small rotary key (64 numbers) shared across heads and cached alongside.</p>`,
    widget: mlaWidget,
    buys: ['A 93.3% smaller cache than DeepSeek 67B, per the paper — about the size of GQA with 2.25 groups.', 'Reported quality stronger than standard multi-head attention.', 'Every head still gets its own keys and values.'],
    costs: ['More projections and a more intricate kernel; more compute per cached byte.', 'Position needs a special decoupled path.', 'The cache still grows linearly with context — per-token storage shrank, the number of tokens did not.'],
    choose: 'Large models where serving memory dominates and you control the kernel stack. DeepSeek-V3 kept it; Kimi Linear uses MLA for its global layers.',
    next: { id: 'nsa', text: 'DeepSeek’s next moves attack how much of that history each query has to read.' },
  },

  'gated-deltanet': {
    oneliner: 'Delta-rule precision, plus a gate that can forget fast.',
    moment: `<p>Two state designs had complementary flaws. Mamba-2’s decay gate can wipe memory quickly but cannot edit one association precisely; DeltaNet edits precisely but changes one key–value pair at a time and cannot clear outdated information at a topic change. A mid-2024 algorithm had finally made DeltaNet trainable in parallel.</p>`,
    change: `<p>Combine them: <code>S ← α S (I − β k kᵀ) + β v kᵀ</code>. The scalar gate α (between 0 and 1) decays the whole state; the delta term then corrects the association for the current key. Trained chunk-wise in parallel.</p>`,
    widget: () => deltaWidget('gated'),
    buys: ['Fast erasure when context changes, targeted edits otherwise.', 'Better language modelling and recall than Mamba-2 or DeltaNet alone in the paper.', 'In production: Qwen3-Next and Qwen3.5 use 3 Gated DeltaNet layers per attention layer; Kimi Linear’s KDA extends it.'],
    costs: ['Still a fixed-size, lossy memory — production models keep softmax layers alongside it.', 'The gate can forget things that were still needed.', 'Complex kernels and more hyperparameters.'],
    choose: 'As the state layer of a hybrid for long-context serving.',
    next: { id: 'nsa', text: 'The other camp keeps softmax everywhere and instead reads less of the history.' },
  },

  nsa: {
    oneliner: 'Learn which blocks to read: a coarse view, selected detail, and a local window.',
    moment: `<p>Sparse attention had been tried for six years, but most methods were applied only at inference, could not be trained end-to-end, or promised speedups that real hardware did not deliver. Long-context reasoning needed 64K tokens and more.</p>`,
    change: `<p>Three branches per query, mixed by learned gates: (1) <strong>compressed</strong> summaries of blocks of tokens, read in full; (2) <strong>selected</strong> blocks, the top-scoring ones according to the compressed scores, read at full detail; (3) a <strong>sliding window</strong> of recent tokens. The block layout is designed around how GPUs move memory, and the model is trained this way from pretraining onwards — not sparsified afterwards.</p>`,
    widget: () => deepseekWidget('nsa'),
    buys: ['Trainable end to end — the selector learns what to select.', 'Reported to match or beat full attention on the paper’s benchmarks, with large speedups at 64K for decoding, forward and backward passes.', 'Gives top-k its cheap proposal step: the compressed branch.'],
    costs: ['All token-level keys and values are still stored: compute falls, the cache does not.', 'Block granularity can miss scattered single tokens.', 'Complex kernel, three branches and gates to tune; tied to shared-KV head layouts.'],
    choose: 'Training a long-context model from scratch when you own the kernel stack.',
    next: { id: 'dsa', text: 'Seven months later DeepSeek ships a simpler, token-level version.' },
  },

  dsa: {
    oneliner: 'A tiny indexer scores everything; the real attention reads 2,048 tokens.',
    moment: `<p>DeepSeek wanted cheaper long-context inference for its existing MLA model, ideally by continued training rather than a new architecture.</p>`,
    change: `<p>A <strong>lightning indexer</strong> — a few small heads, ReLU scores, FP8 arithmetic — scores every earlier token for each query. The main MLA attention then reads only the top 2,048. The V3.2 report trains the indexer first to imitate dense attention, then switches the whole model to sparse training.</p>`,
    widget: () => deepseekWidget('dsa'),
    buys: ['Core attention cost O(T · k) instead of T².', 'Retrofitted onto an existing model with continued training.', 'Token-level rather than block-level selection.'],
    costs: ['The indexer itself is still quadratic in T — just very cheap per score.', 'The KV cache is not reduced.', 'If the indexer misses a key, the main attention never sees it.'],
    choose: 'Cutting long-context inference compute for an existing MLA-style model.',
    next: { id: 'csa', text: 'The next step shrinks the history itself, so both the cache and the reads fall.' },
  },

  drope: {
    oneliner: 'Use RoPE to learn, then take it away.',
    moment: `<p>Extension methods rescale RoPE and still degrade as factors grow; NoPE models generalise in length but are harder to train well. The lesson: a positional function can be defined at 256K while the model’s competence there is unproven.</p>`,
    change: `<p>Gelberg et al. pretrain normally with RoPE, then remove positional embeddings from every layer and continue training briefly at the <em>original</em> context length — no long-context fine-tuning. The model is then evaluated at much longer lengths, relying on the causal mask for order as in NoPE. Their recalibration budgets are small relative to pretraining (for example 20B tokens for Llama-2-7B).</p>`,
    widget: dropeWidget,
    buys: ['Nothing positional left to go out of range.', 'Reported to beat PI, NTK-aware and YaRN baselines on the paper’s long-context benchmarks.', 'Cheap relative to training long natively — the “train short, then stretch” road.'],
    costs: ['Requires an extra training phase — it is not an inference-time switch (it is applied before annealing).', 'Evidence so far is at modest scale (up to 7B in the paper).', 'Says nothing about compute or cache: a long context still needs those bills paid.'],
    choose: 'Extending a pretrained model’s usable context cheaply when a short continued-training phase is affordable. One reported recipe applies DroPE before annealing for an 8K → 256K extension.',
    next: { id: 'csa', text: 'The opposite road — build long, train long — is what DeepSeek’s next architecture takes.' },
  },

  csa: {
    oneliner: 'Compress the history, then read only the compressed blocks that matter.',
    moment: `<p>Targeting a million tokens, even DSA was not enough: it still stores a full-size entry for every token, and at 1M tokens the cache dominates.</p>`,
    change: `<p>Two attention types, interleaved across layers. <strong>Compressed Sparse Attention</strong> merges every 4 tokens into one KV entry, then uses a lightning indexer to pick the top-k compressed entries (1,024 for V4-Pro), plus a 128-token uncompressed window for recent detail. <strong>Heavily Compressed Attention</strong> merges 128 tokens per entry and attends to all of them densely — a cheap global overview. RoPE is applied to the last 64 dimensions .</p>`,
    widget: () => deepseekWidget('csa'),
    buys: ['Both bills at once: fewer entries stored (sequence compression) and fewer read (selection).', 'Per the report, at 1M tokens V4-Pro needs 27% of V3.2’s single-token FLOPs and 10% of its KV cache.', 'Native long-context training rather than stretching.'],
    costs: ['Compression destroys token-level detail; the summaries and the indexer must be learned well.', 'Selection can miss what mattered; a single lab’s evidence so far.', 'Very complex: several attention types, indexers and kernels — the expensive road.'],
    choose: 'Building a million-token model from scratch with full control of training and kernels.',
    next: { text: 'Where the timeline stops: linear-state hybrids (Qwen, Kimi) and compression-plus-selection (DeepSeek) are two live answers to the same bills. Choosing between them is the next-generation design decision.' },
  },
};

export { attentionWidget };
