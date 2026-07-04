"""
Headless test of the Concord engine (no GUI). Uses a heuristic mock aligner
so it runs anywhere; swap to SimAlign for real alignment:

    python cli_test.py sample-test.xlf --real      # uses SimAlign (downloads model)
    python cli_test.py sample-test.xlf             # uses mock heuristic aligner
"""
import sys
from concord.core.xliff import parse_xliff
from concord.core.aligner import MockAligner, CachingAligner, build_aligner
from concord.core.engine import ConsistencyEngine, EngineConfig


class HeuristicAligner(MockAligner):
    """Monotonic proportional alignment — placeholder when no model is available."""
    def align(self, src, tgt):
        if not src or not tgt:
            return []
        return [(i, min(int(i * len(tgt) / len(src)), len(tgt) - 1)) for i in range(len(src))]


def main():
    if len(sys.argv) < 2:
        print("usage: python cli_test.py <file.xlf> [--real]")
        return
    path = sys.argv[1]
    real = "--real" in sys.argv

    xf = parse_xliff(path)
    print(f"Parsed {xf.name}: {len(xf.segments)} segments")

    if real:
        print("Loading SimAlign (mBERT)…")
        aligner = build_aligner("simalign", model="bert")
    else:
        print("Using heuristic mock aligner (install model + use --real for accuracy)")
        aligner = CachingAligner(HeuristicAligner())

    eng = ConsistencyEngine(
        aligner,
        EngineConfig(nmin=2, nmax=3, stop_mode="trim", min_occurrences=2),
    )
    flags = eng.analyze(xf.segments, progress=lambda d, t: None)

    print(f"\nFlagged {len(flags)} inconsistent term(s):\n")
    for f in flags[:20]:
        print(f"  [{f.distinct} spans / {f.total} occ] \"{f.ngram}\"")
        for v in f.variants:
            print(f"       {v.count}× : {v.span}")
    print()


if __name__ == "__main__":
    main()
