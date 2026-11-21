# Report: Post-OCR Correction for Sanskrit / Devanagari

**Track:** Option 3, Post-OCR Correction Model for Sanskrit/Indic Text
**Model:** `google/byt5-small`, fully fine-tuned · **Hardware:** Google Colab T4 (16 GB) · **Notebook:** `notebooks/Sanskrit_Post_OCR_Correction.ipynb`

> Sections marked *(Phase 4)* are completed after the evaluation run.

---

## 1. Problem understanding
OCR engines struggle with Sanskrit in Devanagari script. Glyphs look alike (य/म, व/ठ), vowel signs and the marks above and below a letter are easy to miss, conjuncts (stacked half-letters) break apart, and old print adds noise. The goal is a model that takes **raw OCR text** and returns **corrected text**.

Three properties of the task drove the design:
- **Copy-mostly.** About 90% of characters are already right, so a good corrector must change *only* what's wrong. Over-correction (rewriting correct text) is this task's form of hallucination and the main risk.
- **Character-level, with context.** Most fixes are 1–3 code points, but some can only be decided from the sentence. In *real-word errors*, OCR produces another valid word: रोगः (disease) → योगः (yoga), दोष → द्रोण. A dictionary can't catch these.
- **Hardly any real labelled data.** We have 5 real page images with ground truth, so training data has to be synthetic and the real pages are kept for testing.

## 2. Dataset preparation
**Real test set (held out):** 5 scan-style page images of Ayurvedic Sanskrit with ground-truth text (`Data/`, 705 characters). We OCR'd them with Tesseract 5 (`san`, `--psm 6`, best of a 4-config grid):
**CER 9.6%, WER 33.9%.** This is the "before" number.

**Error analysis drove the data design.** We aligned OCR output to ground truth (51 edits: 38 substitutions, 8 deletions, 5 insertions):
- about 60% look-alike consonant swaps (य→म ×8, व→ठ/त/क/न, ह→द, स→य, ष→ण)
- about 25% mark errors (anusvara `ं`→`े` ×5, dropped visarga `ः` ×3, dropped or reduced vowel signs)
- about 10% conjunct errors (spurious `्र`: दोष→द्रोष; lost half-letters: त्त्व→त्व)
- real-word errors (रोगः→योगः, विहारः→निहार)

**Clean text:** Sanskrit Wikipedia (`wikimedia/wikipedia 20231101.sa`, 12k articles), NFC-normalized and cut into line-sized chunks of 3–10 words. We kept chunks that are ≥90% Devanagari and contain no Latin letters or digits, then removed duplicates. That left **436k clean lines**. Any chunk that shares a 3-word window with a test page was removed (none did).

**Two noise sources:**
1. **Rule-based noise (`src/noise.py`).** A confusion table of look-alike glyphs, mark drops and swaps, conjunct insertions and drops, plus spacing and danda noise, weighted by the error mix above. Noise level is drawn per line, and 10% of lines are left clean. *Calibration:* the synthetic edit mix (sub/del/ins) is **69/17/13** vs real **75/16/10**. Effective CER ≈ rate/2, so training covers roughly 1.5–20% CER.
2. **Render-then-OCR.** 6,000 clean lines drawn with PIL+raqm in 6 Devanagari fonts, then degraded with paper noise, blur, rotation, downscaling and JPEG compression. The **same Tesseract** reads them back, which gives real OCR errors without hand labelling. Tesseract is almost perfect on clean print fonts (CER 0–4%) but not on the handwriting-style **Kalam** (CER ~13–15%), whose style is close to the test pages. Half the renders use Kalam, which brings this set to **10.0% CER** with exactly the real confusions (व→ठ, य→म).

**Splits** (disjoint at the clean-line level; leakage checks are assertions in the notebook):

| split | n | do-nothing CER | already-correct lines |
|---|---|---|---|
| train (rule + render×2) | 67,984 | 7.4% | 24% |
| dev | 1,500 | 8.1% | 23% |
| test_rule | 1,000 | 7.4% | 23% |
| test_render | 500 | 8.8% | 31% |
| real pages | 5 pages | 9.6% | – |

**Unicode normalization:** both sides are NFC-normalized and ZWJ/ZWNJ are removed **before every metric**, so differences in Unicode encoding (e.g. a precomposed nukta letter vs letter + nukta sign) never count as errors. The model's output is NFC-normalized too.

## 3. Why ByT5-small
- **It works on bytes, so it has no vocabulary to fail on.** OCR output contains Latin look-alikes (`|`, `l`, `:`), stray marks and invalid vowel-sign sequences. A byte model can read and write any string, and a correction is a small byte-level edit.
- **We tested the tokenizer argument instead of assuming it.** On 1,000 dev lines, mT5's SentencePiece handled Devanagari *better than expected*: 0% `<unk>`, 99.9% exact round-trip, sensible splits (संयोगः → `▁सं|योग|ः`). A one-character OCR edit changes about 2 tokens for *both* models. So subword fragmentation turned out **not** to be a decisive problem. We still chose ByT5 because it's lossless by design (100% round-trip) and because of the ByT5 paper's reported robustness to noisy and misspelled input.
- **The cost is length:** ByT5 sequences are **6.3× longer** (2.79 tokens per character vs 0.44). That's the main compute cost on a T4 and the reason we trained for only one epoch. **mT5-small is the obvious cheaper ablation** (future work).
- **Why not an LLM (Llama-3.2-1B / Gemma-2B)?** They're much larger for a narrow copy-and-edit task. Decoder-only models also paraphrase more readily, which means more over-correction, and they'd need LoRA/QLoRA to fit on a T4.
- **Tokenizer adaptation:** none needed. ByT5 has no vocabulary, and mT5 showed 0% unknown tokens.

## 4. Fine-tuning approach
Full fine-tuning (300M parameters fit on a T4, so LoRA isn't needed) with the Hugging Face `Seq2SeqTrainer` (`src/train.py`):
- Input is the noisy line and the target is the clean line. `max_length=384` bytes (target p99 285, max 349, so nothing is truncated).
- Adafactor, learning rate 5e-4, linear decay with 3% warmup, effective batch 32 (8 × 4 gradient accumulation), **1 epoch = 2,125 steps**.
- Length-grouped batching (`train_sampling_strategy='group_by_length'`) to cut padding.
- Every 400 steps: evaluate on 300 dev lines **with real generated output** (CER/WER, not just loss) and save a checkpoint to Google Drive. The best checkpoint is kept by dev CER.
- Training is resumable: after a runtime reset, the script continues from the latest Drive checkpoint, including optimizer, scheduler and data position.

Learning curve *(Phase 4: full curve)*: training loss went 3.08 → 0.74 → 0.45 → 0.34 over steps 50–400. Dev CER at step 400 was **7.8%** (do-nothing 8.1%): the model first learns to copy, then to correct.

## 5. Hardware constraints and optimizations
| Constraint | Decision |
|---|---|
| T4 has no bf16, and T5 models overflow to NaN in fp16 | **fp32** training. The script switches to bf16 automatically on L4/A100 (`compute capability ≥ 8`) |
| Batch 16 × 384 bytes ran out of memory in fp32 (the longest batch comes first with length grouping) | Batch 8 × 4 gradient accumulation (same effective batch 32), `expandable_segments` allocator |
| Bytes make sequences 6× longer | Length-grouped batches, short line-level inputs, 1 epoch |
| Measured speed 1.85–3.3 s/step (2 epochs ≈ 2–3 h) | 1 epoch; more epochs under future work |
| Colab runtimes reset or disconnect | Checkpoints and data backups on Drive, resumable training, training run as a **detached process** |
| Rendering + OCR is CPU-bound (2 vCPUs) | Multiprocessing pool, resumable batches of 1,500 |

## 6. Evaluation methodology *(Phase 4)*
- **Metrics:** CER and WER (`jiwer`) after NFC normalization, reported *before* (raw OCR) and *after* (corrected) on three test sets: synthetic rule noise, rendered + real Tesseract, and the **5 real pages**, which is the result that matters most.
- **Over-correction rate:** the share of already-correct lines the model changed. This measures hallucination.
- **Fully-fixed rate:** the share of broken lines restored exactly. **Made-worse rate:** lines whose CER went up.
- **Error-type breakdown:** remaining edits grouped as consonant / mark / conjunct / space-punct, before vs after.
- **Real pages:** 95% bootstrap confidence intervals over lines, because 705 characters is a tiny sample.
- **Decoding:** greedy vs beam search (4 beams).
- **test_render by font**, to measure how much the Kalam font-matching helps.

*Results table: TBD.*

## 7. Failure cases *(Phase 4)*
*TBD: qualitative before/after examples, including real-word errors and over-corrections.*

## 8. Challenges encountered
- **Changed library APIs:** Colab ships `transformers` 5.x, where `warmup_ratio` was folded into `warmup_steps` (a float means a ratio) and `group_by_length` became `train_sampling_strategy`. We inspected the signatures instead of guessing and fixed both together.
- **Running out of GPU memory** on the first step. Length grouping deliberately puts the longest batch first, so we fixed it at the worst case with gradient accumulation.
- **Tesseract was too good on clean fonts:** naive rendering gave about 2% CER, which teaches little. We measured per font and weighted toward the font that produces realistic errors, and we report that as a possible source of optimistic results.
- **Notebook and runtime instability:** the first background launch (`!nohup … &`) stayed attached to the cell and blocked the notebook. After a browser reconnect, training stalled at step 400. Because checkpoints and data were on Drive, we resumed from `checkpoint-400` with no loss, using a properly detached `subprocess.Popen(start_new_session=True)`.
- **The tokenizer hypothesis was wrong:** we expected mT5 to fragment Sanskrit badly, and measured that it doesn't. We kept ByT5 for other, stated reasons and documented the finding.

## 9. What we would improve with more time
1. **More realistic data:** classical corpora (GRETIL, DCS, Bhagavad Gita, Ayurvedic texts) instead of Wikipedia, real scanned pages with aligned text, and more fonts and degradations (bleed-through, broken glyphs).
2. **Correlated noise:** errors that cluster per word and per line, and confusion tables learned from OCR runs instead of written by hand.
3. **Ablations:** mT5-small vs ByT5-small (speed and quality), LoRA vs full fine-tuning, 1 vs 3 epochs, and with vs without render pairs.
4. **Confidence-gated correction:** apply an edit only when the model's confidence is high, using token log-probabilities or agreement between beams, to push over-correction toward zero.
5. **Page-level context:** correct with neighbouring lines as context to resolve more real-word errors. Add IAST/Roman transliteration noise.
6. **Deployment:** int8 quantization or ONNX for CPU inference, plus a simple OCR → correct → diff-view web demo for human review.
