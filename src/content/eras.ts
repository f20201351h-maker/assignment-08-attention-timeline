// Era interludes. Each era starts at the first entry whose id is listed; the
// span label is computed from the ledger dates of its entries.

export interface Era { id: string; first: string; title: string; body: string }

export const ERAS: Era[] = [
  {
    id: 'era-prologue', first: 'additive', title: 'Before the Transformer: attention as a helper',
    body: 'Attention starts life as an add-on to recurrent translators, invented to escape a fixed-size memory. Keep that in mind: much later, the field will go back to fixed-size memory on purpose.',
  },
  {
    id: 'era-exact', first: 'sdpa', title: 'Everything sees everything',
    body: 'The Transformer makes attention the whole model. The priority is exactness and parallel training: every token compares itself with every other, exactly. Position has to be bolted on, and the first attempts are absolute.',
  },
  {
    id: 'era-quadratic', first: 'transformer-xl', title: 'The quadratic bill arrives',
    body: 'Two years in, the T² cost is the obvious pain, and ideas arrive quickly: carry state between windows, compute only some pairs, keep only the top-k keys, attend locally, drop softmax altogether. One quieter paper notices the other bill — decoding is bound by cache bandwidth. Almost none of these ideas go into flagship models yet.',
  },
  {
    id: 'era-rethink', first: 'delta', title: 'Rethinking memory and position',
    body: 'Two threads mature. Fixed-size memories get a better write rule. Position stops being a vector added to the input and moves inside attention — as a rotation, or as a penalty on distance.',
  },
  {
    id: 'era-exact-again', first: 'nope', title: 'Exactness strikes back',
    body: 'A kernel paper changes the economics: exact attention, computed in tiles, becomes fast and memory-light — the approximations had often failed to deliver real speedups. Meanwhile it turns out a causal model barely needs positional encodings, and state-space models start borrowing a few attention layers.',
  },
  {
    id: 'era-serving', first: 'gqa', title: 'Serving, and stretching',
    body: 'LLMs are in production, and the bill that hurts is the cache, per user, per token. Anything that cuts the cache is now adopted within weeks — MQA’s idea, waiting since 2019, returns as GQA. In about ten weeks of one summer, a hobbyist’s blog, a paper, a forum post and another paper race to stretch RoPE past its training length. Streaming exposes the attention sink. And fixed-size state comes back, now selective.',
  },
  {
    id: 'era-compressed', first: 'mla', title: 'Memory, compressed',
    body: 'Two ways to make memory smaller without simply forgetting: store a compact latent instead of full keys and values, or make the fixed-size state smart enough to correct and forget on purpose. Both reach production.',
  },
  {
    id: 'era-selection', first: 'nsa', title: 'Selection becomes native',
    body: 'Sparsity returns after six years, now trained in from the start with a learned selector, then combined with compression of the sequence itself. Position comes full circle: train with RoPE, then remove it.',
  },
];
