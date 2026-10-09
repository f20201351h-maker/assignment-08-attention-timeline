import { h } from './lib/dom.ts';
import { ENTRIES, LEDGER, byId, fmtDate, monthYear, year, yearsBetween } from './lib/ledger.ts';

const lagRow = (id: string) => {
  const e = byId(id);
  const u = e.firstUse!;
  const lag = yearsBetween(e.date, u.date);
  return h('tr', null,
    h('td', null, h('a', { href: `#${e.id}` }, e.short)),
    h('td', { class: 'date' }, monthYear(e.date)),
    h('td', null, u.model),
    h('td', { class: 'date' }, monthYear(u.date)),
    h('td', { class: 'date' }, lag < 0.1 ? 'same time' : lag < 1 ? `${Math.round(lag * 12)} months` : `${lag.toFixed(1)} years`));
};

function whatItShows() {
  const lag = (id: string) => yearsBetween(byId(id).date, byId(id).firstUse!.date);
  const early = ['mqa', 'sliding-window', 'linear', 'delta', 'topk'];
  const avgEarly = early.reduce((a, id) => a + lag(id), 0) / early.length;
  const sec = h('section', { class: 'section', id: 'what-it-shows' });
  sec.append(
    h('div', { class: 'kicker' }, 'Reading the order'),
    h('h2', null, 'What the timeline shows that a list cannot'),
    h('p', null, 'Written after the dates were verified, from what the ordered data actually shows.'),
    h('h3', null, '1. Ideas wait for their bill'),
    h('p', null, `Most of the ideas that save compute or memory were published early — multi-query attention, top-k selection, sliding windows, linear attention and the delta rule all appeared between ${year(byId('mqa').date)} and ${year(byId('delta').date)}. Almost none went into a flagship model at the time. On average those five waited ${avgEarly.toFixed(1)} years for their first notable use. Then grouped-query attention was in Llama 2 ${Math.round(lag('gqa') * 12)} months after publication. The ideas did not get better in between; the bill changed. Once models were served to millions of users, the per-user cache became the cost that hurt, and anything that cut it was adopted almost immediately. As a list, these look like a menu. In date order, they look like a queue.`),
    h('div', { class: 'tbl-wrap' }, h('table', null,
      h('thead', null, h('tr', null, h('th', null, 'Mechanism'), h('th', null, 'Published'), h('th', null, 'First notable use'), h('th', null, 'When'), h('th', null, 'Wait'))),
      h('tbody', null, ...['sparse-transformer', 'mqa', 'topk', 'sliding-window', 'linear', 'delta', 'rope', 'alibi', 'gqa', 'ntk', 'yarn', 'sinks', 'mamba', 'mla', 'gated-deltanet'].map(lagRow)))),
    h('p', { class: 'muted', style: 'font:400 .85rem/1.5 var(--sans)' }, '“First notable use” is the earliest released model I could verify from a primary source; an earlier use I missed would shorten a wait, never lengthen it. Delta rule and top-k are counted at their first use inside a larger design (Gated DeltaNet layers; DeepSeek’s indexer).'),
    h('h3', null, '2. The priorities really do rotate — but compute came first'),
    h('p', null, 'The common summary — first exactness, then memory, then length, then memory again — holds up, with one addition the dates make obvious. After the exact, everything-sees-everything Transformer, the first wave (2019–2020) was mostly about compute: sparse patterns, windows, linear attention (top-k was pitched as a quality fix, and MQA was already about the cache, unheeded). Then a kernel, FlashAttention, made exact attention fast in practice — in my reading, removing much of the practical case for approximations at moderate lengths. Only after that did memory (MQA’s idea finally adopted as GQA, then MLA) and length (PI, NTK-aware, YaRN within ten weeks of each other) take over, followed by memory again in a new form (selective and delta-rule state, latent caches, compressed sequences).'),
    h('h3', null, '3. The field went back to the thing attention was invented to escape'),
    h('p', null, `Attention was born in ${year(byId('additive').date)} to get rid of a fixed-size summary of the past. In ${year(byId('linear').date)} linear attention deliberately reintroduced one, for cost — and the next five years were spent making that fixed state smarter: correct-before-write (delta rule), content-dependent forgetting (Mamba), both together (Gated DeltaNet). It never fully won: the production models on this timeline that use fixed state (Jamba, MiniMax-01, Qwen3-Next, Kimi Linear) all keep some exact attention layers. The real answer turned out to be a mixture across depth, not a winner.`),
    h('h3', null, '4. Position swings like a pendulum'),
    h('p', null, `Learned table (${year(byId('learned-abs').date)}) → computed sinusoids (${year(byId('sinusoidal').date)}) → relative terms (${year(byId('relative').date)}) → rotation and distance penalties (${year(byId('rope').date)}) → nothing at all (${year(byId('nope').date)}) → stretching the rotation (${year(byId('pi').date)}) → training with rotation and then removing it (${year(byId('drope').date)}). Each step reacts to the previous one’s failure to work beyond the training length, and the last step only makes sense because of a 2022 result most lists would never include: a causal model can infer order from its mask.`),
    h('h3', null, '5. Old ideas come back with a missing piece'),
    h('p', null, `Top-k selection appeared in ${year(byId('topk').date)} and still scored every key, so it saved little. It returned in ${year(byId('nsa').date)} with the piece it lacked — a cheap, trained selector — and became DeepSeek’s production design. The delta rule is from 1960 and became practical in production only with a parallel training algorithm (${year(byId('gated-deltanet').date)} era). Seen as a list, these are separate techniques; on a timeline, they are the same idea arriving twice, the second time with its missing part.`),
    h('h3', null, 'What I would guess comes next'),
    h('p', null, 'Speculation, labelled as such: the two live camps — linear-state hybrids (Qwen, Kimi) and compression-plus-selection (DeepSeek) — attack the same bills from opposite ends, so the obvious next move is to combine them: learned selection or compression applied to the few exact-attention layers inside state hybrids, with the layer ratio itself chosen by measurement rather than habit. Position will keep moving from a formula into the training procedure.'),
    h('h3', null, 'A mechanism worth adding'),
    h('p', null, 'The most consequential omission is ', h('a', { href: '#flash' }, 'FlashAttention'), ` (Dao et al., arXiv:2205.14135, v1 ${fmtDate(byId('flash').date)}). It is not a new attention mechanism at all — the output is exact — yet, in my reading, it changed which bill mattered and helps explain why so many efficient approximations faded. The other additions (marked on their cards) are there because a later mechanism makes no sense without them: NoPE for DroPE, Position Interpolation for NTK-aware scaling and YaRN, Mamba for Gated DeltaNet, hybrids for the depth schedules.`),
  );
  return sec;
}

function workloads() {
  const sec = h('section', { class: 'section', id: 'choose' });
  sec.append(
    h('div', { class: 'kicker' }, 'Same mechanisms, different workloads'),
    h('h2', null, 'Right for a 2K chatbot, wrong for a 1M agent'),
    h('p', null, 'No mechanism is good or bad in isolation. A sketch of reasonable defaults today, assuming you are building a decoder model:'),
    h('div', { class: 'choices' },
      h('div', { class: 'choice' }, h('h4', null, '2K–8K chatbot'),
        h('ul', null,
          h('li', null, 'Plain softmax attention with FlashAttention: T² is cheap at this length and exact recall is free.'),
          h('li', null, 'RoPE; GQA to cut the per-user cache for high concurrency.'),
          h('li', null, 'Skip linear/state layers and sparse selection: their complexity buys nothing here.'))),
      h('div', { class: 'choice' }, h('h4', null, '128K documents / RAG'),
        h('ul', null,
          h('li', null, 'Cache is the bill: GQA at minimum, MLA if you control the kernels.'),
          h('li', null, 'Train at a shorter length and extend with YaRN-style scaling, or recalibrate (DroPE-style) — then measure retrieval at the target length, not just perplexity.'),
          h('li', null, 'Interleaved sliding-window and global layers cut cost if most layers can stay local.'))),
      h('div', { class: 'choice' }, h('h4', null, '1M-token agent'),
        h('ul', null,
          h('li', null, 'GQA alone is not enough: the cache still grows linearly.'),
          h('li', null, 'Either fixed-state hybrids (Gated DeltaNet with a few attention layers) or sequence compression plus learned selection (DeepSeek-V4 style).'),
          h('li', null, 'Native long-context training, and evaluation that tests reasoning across the context — a needle test alone proves little.')))),
    h('p', null, 'Two roads apply on top: ', h('strong', null, 'train short, then stretch'), ' (cheap, with a ceiling — e.g. an 8K → 256K extension with DroPE) or ', h('strong', null, 'build long, then train long'), ' (expensive, no stretching ceiling — DeepSeek’s compressed and sparse attention). GQA, sparse attention and better position schemes serve either road.'),
  );
  return sec;
}

function courseNotes() {
  const item = (title: string, said: string, body: string) => h('div', { class: 'erratum' }, h('h4', null, title), h('p', { class: 'said' }, said), h('p', { html: body }));
  const sec = h('section', { class: 'section', id: 'notes' });
  sec.append(
    h('div', { class: 'kicker' }, 'Checking the notes I learned from'),
    h('h2', null, 'What I confirmed, and two things worth refining'),
    h('p', null, 'I checked the claims in my course notes (ERA V5, Session 8) that touch dates and mechanisms against primary sources. Most held up exactly; two deserve an update.'),
    h('div', { class: 'errata' },
      item('Refine: the DroPE algorithm is public', 'Notes: the available record does not establish the exact DroPE algorithm or which rotary dimensions it changes.',
        `True of the record the notes rely on, but the method itself is published: Gelberg, Eguchi, Akiba & Cetin, “Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings” (arXiv:2512.12167, ${fmtDate(byId('drope').date)}). It drops RoPE from <em>every</em> layer after pretraining and recalibrates briefly at the original context length. The notes’ caution — a recalibration step, not an inference-time switch — matches the paper.`),
      item('Refine: a causal decoder is not a pure set-reader', 'Notes: attention itself does not know token order; the transformer reads a set.',
        `Exactly right for attention scores and for bidirectional encoders. For a <em>causal</em> decoder, the mask itself leaks order: token t sees t tokens and can infer its position. Haviv et al. (arXiv:2203.16634, ${fmtDate(byId('nope').date)}) show causal LMs without positional encodings remain competitive. This is why DroPE can work at all, so it is worth a footnote.`),
      item('Confirmed: MQA before GQA', 'Notes: MQA is introduced as the one-K/V-head extreme of GQA.',
        `The right way to teach it; historically the order is reversed — MQA (${monthYear(byId('mqa').date)}) came three and a half years before GQA (${monthYear(byId('gqa').date)}), which was proposed as the middle ground.`),
      item('Confirmed: DeepSeek-V4 details', 'Notes: RoPE on the last 64 dimensions; compressed blocks, a low-rank indexer selecting top-k, and a heavily compressed dense form interleaved.',
        'All confirmed by the DeepSeek-V4 technical report (arXiv:2606.19348): CSA compresses every 4 tokens and selects top-k with a lightning indexer; HCA compresses 128 tokens per entry and attends densely; RoPE is applied to the last 64 dimensions.'),
      item('Confirmed: the cache yardstick', 'Notes: 48 layers, 8 KV heads, head dim 128, bf16, 32,768 tokens → ≈ 6.44 GB per user, ≈ 51.54 GB for eight.',
        'Reproduced exactly by this site’s cache formula (and checked in its test suite).'),
    ),
  );
  return sec;
}

function sources() {
  const sec = h('section', { class: 'section', id: 'sources' });
  sec.append(
    h('div', { class: 'kicker' }, 'Method'),
    h('h2', null, 'How the dates were checked'),
    h('p', null, LEDGER.dateRule),
    h('p', null, 'Process: a first research pass fetched every primary page; an independent second pass, told to distrust the first, re-fetched each one and hunted for earlier appearances. That audit moved one date (learned absolute positions, back six months to an earlier paper by the same group), added earlier roots to several entries, and resolved the DeepSeek-V4 report’s listing. Every date on this page is read from one machine-readable ledger (src/data/ledger.json) — the build fails if the page and the ledger disagree.'),
    h('p', null, LEDGER.effectsKey),
    h('div', { class: 'tbl-wrap' }, h('table', null,
      h('thead', null, h('tr', null, h('th', null, 'Date'), h('th', null, 'Mechanism'), h('th', null, 'What the date is'), h('th', null, 'Primary source'))),
      h('tbody', null, ...ENTRIES.map((e) => h('tr', null,
        h('td', { class: 'date' }, fmtDate(e.date)),
        h('td', null, h('a', { href: `#${e.id}` }, e.short)),
        h('td', null, e.dateEvent),
        h('td', null, h('a', { href: e.source.url, target: '_blank', rel: 'noopener' }, e.source.id))))))),
    h('p', { class: 'muted', style: 'font:400 .85rem/1.5 var(--sans)' }, 'Known soft spots, stated rather than hidden: the NTK-aware Reddit post’s timestamp was read from an archive mirror because reddit.com blocked automated fetching; the date of Jianlin Su’s original RoPE blog posts (early 2021) could not be fetched; GPT-1’s release date is as reported by search because openai.com blocked automated fetching. None of these change the order of the timeline.'),
  );
  return sec;
}

export function buildSections() {
  return [whatItShows(), workloads(), courseNotes(), sources(),
    h('footer', null,
      h('p', null, 'Built for ERA V5 Session 8 (The School of AI). Interactive figures use small hand-made toy numbers unless a caption says otherwise; formulas and reported figures come from the cited papers.'),
      h('p', null, 'Every date links to its primary source. If you find one that is wrong, that is exactly the kind of correction this page is meant to invite.'))];
}
