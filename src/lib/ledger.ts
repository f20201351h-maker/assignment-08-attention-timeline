import raw from '../data/ledger.json';

export type Pressure = 'compute' | 'memory' | 'position' | 'fidelity';
export interface Ref { title: string; date: string; url: string; note?: string }
export interface Use { what: string; date: string; url: string; note?: string }
export interface Entry {
  id: string; name: string; short: string; date: string; dateEvent: string; kind: string;
  tier: 'milestone' | 'connective'; inClassList: boolean; primary: Pressure;
  effects: Record<Pressure, number>;
  source: { title: string; authors: string; id: string; url: string; venue: string };
  precursors: Ref[]; adoption: Use[]; notes: string; verification: string;
  firstUse?: { model: string; date: string; url: string };
}

export const LEDGER = raw as unknown as { dateRule: string; effectsKey: string; entries: Entry[] };
/** Timeline order = date order; the ledger file order breaks same-day ties. */
export const ENTRIES: Entry[] = [...LEDGER.entries].sort((a, b) => a.date.localeCompare(b.date));
export const byId = (id: string) => {
  const e = ENTRIES.find((x) => x.id === id);
  if (!e) throw new Error(`unknown entry ${id}`);
  return e;
};

export const PRESSURES: { key: Pressure; label: string; blurb: string }[] = [
  { key: 'compute', label: 'Compute', blurb: 'the T² bill: query–key work' },
  { key: 'memory', label: 'Memory', blurb: 'the KV-cache / state bill' },
  { key: 'position', label: 'Position', blurb: 'order and reach beyond training length' },
  { key: 'fidelity', label: 'Fidelity', blurb: 'exact access to any earlier token' },
];

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** "2021-04-20" -> "20 Apr 2021"; "2021-03" -> "Mar 2021"; "1992" -> "1992". */
export function fmtDate(d: string): string {
  const [y, m, day] = d.split('-');
  if (!m) return y;
  if (!day) return `${MONTHS[Number(m) - 1]} ${y}`;
  return `${Number(day)} ${MONTHS[Number(m) - 1]} ${y}`;
}
export const monthYear = (d: string) => { const [y, m] = d.split('-'); return m ? `${MONTHS[Number(m) - 1]} ${y}` : y; };
export const year = (d: string) => d.slice(0, 4);
/** Fractional year for plotting. */
export const fyear = (d: string) => { const [y, m = '6', day = '15'] = d.split('-'); return Number(y) + (Number(m) - 1) / 12 + (Number(day) - 1) / 365; };

/** Years between two ISO-ish dates. */
export const yearsBetween = (a: string, b: string) => fyear(b) - fyear(a);
