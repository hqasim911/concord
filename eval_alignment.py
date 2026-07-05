"""
Alignment evaluation harness — precision / recall / AER against a gold set.

    python eval_alignment.py                      # heuristic mock aligner
    python eval_alignment.py --real               # SimAlign (mBERT)
    python eval_alignment.py --backend awesome    # awesome-align
    python eval_alignment.py --backend ensemble   # SimAlign ∩ awesome-align
    python eval_alignment.py --gold eval/gold.tsv --model xlmr

AER (Alignment Error Rate, lower is better):
    AER = 1 - (|A∩S| + |A∩P|) / (|A| + |S|)
where A = system links, S = sure gold links, P = sure ∪ possible.
"""
import sys

from concord.core.aligner import build_aligner, CachingAligner, MockAligner


class HeuristicAligner(MockAligner):
    """Monotonic proportional alignment — baseline needing no model."""
    def align(self, src, tgt):
        if not src or not tgt:
            return []
        return [(i, min(int(i * len(tgt) / len(src)), len(tgt) - 1))
                for i in range(len(src))]


def load_gold(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            src, tgt, links = parts[0], parts[1], parts[2]
            sure, poss = set(), set()
            for tok in links.split():
                if "?" in tok:
                    i, j = tok.split("?")
                    poss.add((int(i), int(j)))
                elif "-" in tok:
                    i, j = tok.split("-")
                    sure.add((int(i), int(j)))
            rows.append((src.split(), tgt.split(), sure, sure | poss))
    return rows


def evaluate(aligner, rows):
    a_and_s = a_and_p = a_tot = s_tot = 0
    for src, tgt, sure, poss in rows:
        a = set(aligner.align(src, tgt))
        a_and_s += len(a & sure)
        a_and_p += len(a & poss)
        a_tot += len(a)
        s_tot += len(sure)
    precision = a_and_p / a_tot if a_tot else 0.0
    recall = a_and_s / s_tot if s_tot else 0.0
    aer = 1 - (a_and_s + a_and_p) / (a_tot + s_tot) if (a_tot + s_tot) else 1.0
    return precision, recall, aer


def main():
    args = sys.argv[1:]
    gold = "eval/gold.tsv"
    if "--gold" in args:
        gold = args[args.index("--gold") + 1]
    model = "bert"
    if "--model" in args:
        model = args[args.index("--model") + 1]
    backend = None
    if "--backend" in args:
        backend = args[args.index("--backend") + 1]
    elif "--real" in args:
        backend = "simalign"

    rows = load_gold(gold)
    print(f"Gold: {gold} — {len(rows)} sentence pairs\n")

    if backend:
        print(f"Loading backend '{backend}' (model={model})…")
        aligner = build_aligner(backend, model=model)
        name = backend
    else:
        aligner = CachingAligner(HeuristicAligner())
        name = "heuristic (baseline)"

    p, r, aer = evaluate(aligner, rows)
    print(f"\nBackend : {name}")
    print(f"Precision : {p:.3f}")
    print(f"Recall    : {r:.3f}")
    print(f"AER       : {aer:.3f}   (lower is better)")


if __name__ == "__main__":
    main()
