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
    aligner.py    # swappable alignment: SimAlign (neural) | Mock | Caching wrapper
    textutil.py   # Arabic normalization, tokenization, n-gram extraction
    xliff.py      # XLIFF 1.2 / 2.0 parse + round-trip editing (lxml)
    engine.py     # the consistency engine (n-gram -> aligned span -> grouping)
    llm.py        # optional, provider-agnostic LLM verdicts
  api.py          # pywebview backend bridge
  ui/             # desktop frontend (index.html + app.js)
  app.py          # entry point
run.py
cli_test.py       # headless engine test (no GUI)
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads the alignment model (~700MB for mBERT) from Hugging Face.

## Run the desktop app

```bash
python run.py
```

1. **Load model** — pick mBERT (faster) or XLM-R (often better on Arabic), click Load.
2. **Open XLIFF files** — select one or many `.xlf` / `.xliff`.
3. Set **n-gram length**, **stopword mode**, **min occurrences**, **Arabic normalization**.
4. **Analyze** — progress bar shows alignment over all segments.
5. Review flagged terms; each shows the distinct **aligned Arabic spans**. Edit any
   target field (RTL), use **Use for all** to standardize, **↺** to revert.
6. **Export corrected XLIFF** — writes one `*-corrected.xlf` per edited file.

## Headless test (no GUI)

```bash
python cli_test.py path/to/file.xlf          # heuristic mock aligner (fast, rough)
python cli_test.py path/to/file.xlf --real   # real SimAlign (downloads model)
```

## Optional LLM layer

In the app, expand **"connect an LLM"** and paste a base URL, API key, and model.
The client speaks the OpenAI-compatible `/v1/chat/completions` shape, which works
with OpenAI, Together, Groq, OpenRouter, local Ollama (`http://localhost:11434/v1`),
and similar. Per flagged group, **LLM check** asks for a verdict
(inconsistent vs. acceptable variant) and a preferred translation.

> Note: Anthropic's native API uses a different request shape (`/v1/messages`).
> To use Claude directly, point the base URL at an OpenAI-compatible proxy, or
> tell me and I'll add a native Anthropic adapter to `llm.py`.

## Performance notes

- Neural alignment is the slow step. mBERT on CPU handles low-thousands of
  segments in a few minutes; XLM-R is slower but sharper.
- Alignments are cached per unique segment, so re-analyzing with a different
  n-gram range does **not** re-run the model — it's near-instant.
- A CUDA GPU is used automatically if `torch` sees one.

## Swapping the aligner

`build_aligner("simalign", model="bert")` is the default. The `Aligner` interface
in `aligner.py` is one method (`align(src_tokens, tgt_tokens) -> [(i,j)]`), so
awesome-align or an API-based aligner can drop in without touching the engine.
