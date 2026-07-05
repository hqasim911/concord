# Concord — Aligned XLIFF Consistency Checker

Finds English source phrases that were translated **inconsistently** into Arabic —
by aligning each English n-gram to the **exact Arabic span** that translates it,
then comparing those spans. Unlike whole-segment comparison, this does not flag a
term as inconsistent just because the rest of the segment differs.

**Example it gets right:**
- `Click the Save button to continue.` → `انقر على زر حفظ للمتابعة.`
- `Click the Save button twice.` → `انقر على زر حفظ مرتين.`

Whole-segment comparison flags "click the save button" (the targets differ).
Concord aligns the n-gram to **انقر على زر حفظ** in both → correctly **consistent**.

---

## What's inside

```
concord/
  core/
    aligner.py    # SimAlign | awesome-align | Ensemble | Mock | Caching
    textutil.py   # Arabic normalize + light-stem, n-grams, span trim, edit dist
    xliff.py      # XLIFF 1.2 / 2.0 parse, placeholder QA, safe round-trip edit
    engine.py     # consistency engine: forward + reverse, clustering, scoring
    glossary.py   # termbase (CSV/TBX) adherence checking
    llm.py        # optional LLM verdicts (OpenAI-compatible + native Anthropic)
  api.py          # pywebview backend bridge
  ui/             # desktop frontend (index.html + app.js)
  app.py          # entry point
run.py
cli_test.py       # headless engine test (no GUI)
eval_alignment.py # alignment precision/recall/AER harness
eval/gold.tsv     # hand-aligned gold set
sample-test.xlf   # sample fixture
requirements.txt
```

## Setup

Install into a **fresh virtual environment** — not a shared base/conda
environment. The pinned `transformers`/`tokenizers` versions can collide with
other packages otherwise, which surfaces as a `tokenizers ... is required`
import error when the model loads.

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1            # cmd.exe: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install before launching the GUI:

```bash
python -c "import simalign, transformers; print('deps OK')"
```

Notes:
- First run downloads the alignment model (~700MB for mBERT) from Hugging Face.
- On Windows, `torch` from PyPI installs the CPU build (no CUDA index URL
  needed). For GPU, install the matching CUDA wheel per pytorch.org first.
- On Windows, pywebview renders via the **Edge WebView2 Runtime** — bundled
  with Windows 11 and recent Windows 10; otherwise a free download from
  Microsoft.
- To leave the environment later: `deactivate`.

## Run the desktop app

```bash
python run.py
```

1. **Load model** — pick a backend (**SimAlign**, **awesome-align**, or
   **Ensemble** = the intersection of both for higher precision) and a base
   model (mBERT faster, XLM-R often better on Arabic), click Load.
2. **Open XLIFF files** — select one or many `.xlf` / `.xliff`.
3. Set **n-gram length**, **stopwords**, **min occurrences**, **Arabic
   normalization**, **article/clitic folding**, **near-duplicate merging**,
   **min variant count**, and **direction** (EN→AR, or also AR→EN).
4. **Analyze** — each flag shows its distinct aligned spans and a **% split**
   inconsistency score (higher = more evenly divided). Edit targets (RTL),
   **Use for all** to standardize, **↺** to revert.
5. **Tools row** — switch to the **Reverse** view (over-loaded Arabic spans),
   **Load glossary** + **Check adherence**, click **Placeholder issues** to
   list source/target placeholder mismatches, or **LLM-check all**.
6. **Export corrected XLIFF** — writes one `*-corrected.xlf` per edited file
   (reports if any inline tags had to be dropped).

## Accuracy & QA features

- **Article / clitic folding** — collapses الزر / بالزر / وبالزر → زر so
  grammatical variation isn't mistaken for inconsistency (conservative: only
  article-bearing prefixes).
- **Span-outlier trimming** — a single mis-aligned link can't balloon a span
  to swallow unrelated words; only the densest aligned cluster is kept.
- **Near-duplicate clustering + score** — spans within a small edit distance
  merge before counting distinctness; each flag gets an entropy-based score so
  genuine 50/50 splits rank above "one dominant term + noise".
- **Reverse check** — flags one Arabic span used for several English terms
  (over-loaded / ambiguous target term). Enable "+ AR→EN".
- **Glossary adherence** — load a termbase (CSV `source,target` or basic TBX)
  and flag segments that don't use the approved translation.
- **Placeholder QA** — reports segments whose target placeholder set (`%s`,
  `{0}`, `<ph/>`, …) differs from the source's. Editing no longer silently
  drops inline tags.
- **Ensemble aligner** — `build_aligner("ensemble")` intersects SimAlign and
  awesome-align.

## Headless test (no GUI)

```bash
python cli_test.py path/to/file.xlf          # heuristic mock aligner (fast, rough)
python cli_test.py path/to/file.xlf --real   # real SimAlign (downloads model)
```

## Alignment evaluation (AER)

Measure alignment quality against the gold set before trusting a backend:

```bash
python eval_alignment.py                      # heuristic baseline
python eval_alignment.py --real               # SimAlign (mBERT)
python eval_alignment.py --backend awesome    # awesome-align
python eval_alignment.py --backend ensemble   # SimAlign ∩ awesome-align
```

Reports precision / recall / **AER** (lower is better). On the bundled gold
set, SimAlign scores ~0.25 AER vs ~0.61 for the heuristic baseline.

## Optional LLM layer

In the app, expand **"connect an LLM"**, choose a **provider**, and paste a base
URL, API key, and model.

- **OpenAI-compatible** (`/v1/chat/completions`) — OpenAI, Together, Groq,
  OpenRouter, local Ollama (`http://localhost:11434/v1`), and most proxies.
- **Anthropic** — native `/v1/messages`; point the base URL at
  `https://api.anthropic.com` and use Claude directly (no proxy needed).

Auto-detect picks the shape from the base URL. Per flagged group, **LLM check**
asks for a verdict (inconsistent vs. acceptable variant) and a preferred
translation; **LLM-check all** runs the whole batch concurrently.

## Performance notes

- Neural alignment is the slow step. mBERT on CPU handles low-thousands of
  segments in a few minutes; XLM-R is slower but sharper.
- Each **unique** sentence pair is aligned only once per run (translation
  memories repeat heavily), and alignments are cached, so re-analyzing with
  different n-gram / normalization settings is near-instant.
- A CUDA GPU is used automatically if `torch` sees one.

## Swapping the aligner

`build_aligner("simalign", model="bert")` is the default. Backends: `"simalign"`,
`"awesome"` (the awesome-align extraction method implemented directly on
`transformers`; no extra dependency), and `"ensemble"` (intersection of both for
higher precision). Each accepts `model="bert"`/`"xlmr"`, or a full Hugging Face
id — e.g. a fine-tuned awesome-align checkpoint — for better accuracy.

The `Aligner` interface in `aligner.py` is `align(src, tgt) -> [(i,j)]` plus an
optional `align_batch(pairs)`, so an API-based or batching aligner can drop in
without touching the engine.
