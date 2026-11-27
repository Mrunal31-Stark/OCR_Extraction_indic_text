# The project explained: what I built, why, and what I learned

This is a plain-language walkthrough of the whole project: why I chose this track, every main decision and the alternative I rejected, and what the results mean. The formal write-up with all numbers is in [`REPORT.md`](REPORT.md).

- Model: https://huggingface.co/MrunalYewatkar45/byt5-small-sanskrit-post-ocr
- Code, data, results: https://github.com/Mrunal31-Stark/OCR_Extraction_indic_text

---

## 1. The task in one paragraph
OCR engines misread Sanskrit in Devanagari script. Letters look alike (य/म, व/ठ), small marks above and below a letter (vowel signs, anusvara `ं`, visarga `ः`) get dropped or swapped, and stacked half-letters (conjuncts like त्र, स्य) fall apart. I built a model that takes **the text an OCR engine produced** and returns **the corrected text**. It's a second step that can sit after any OCR engine.

```
page image ──► Tesseract OCR ──► "योगः नाम दोष वेषम्म कारणम्" ──► my ByT5 corrector ──► "योगः नाम दोष वेषम्य कारणम्"
```

## 2. Why I chose Option 3 and not Options 1 or 2
The assignment offered three tracks. I picked the one where I could **prove** progress on real data in 1–2 days on a free T4 GPU:

1. **It's objectively measurable.** Each corrected line can be compared character by character with the true text (CER/WER). Option 1 (an LLM answering, translating and explaining Sanskrit) needs human judgment to grade, and automatic translation scores are unreliable for Sanskrit. Option 2 (retrieval) needs someone to label which passages are relevant to which query.
2. **Real test data came with it.** The assignment supplied 5 scanned-style pages with their true text. That let me test on *real* OCR mistakes, not only on data I generated myself.
3. **Training data is effectively free.** Take any correct Sanskrit text, damage it the way OCR would, and train the model to undo the damage. No labelling is needed. Option 1 would have needed thousands of hand-written or LLM-generated instruction pairs, and Option 2 needs aligned Sanskrit–English text, which is scarce.
4. **It fits the hardware.** A 300M-parameter sequence-to-sequence model can be fully fine-tuned on a T4 in about 1.5 hours. A 1–3B LLM for Option 1 would need QLoRA, and evaluating it by generating long answers is slow.
5. **There's an honest baseline.** "Do nothing", meaning keep the raw OCR, is a real baseline. Any improvement or harm is directly visible, including the model *breaking* correct text.
6. **It covers what the brief cares about:** data generation strategy, noise modelling, Unicode normalization, script-specific problems and tokenizer behaviour.
7. **It's practically useful.** Digitizing Sanskrit manuscripts is bottlenecked by OCR quality, and a corrector improves any OCR engine without retraining it.

**The risk I accepted:** synthetic noise might not look like real OCR errors. I reduced that risk by measuring real Tesseract errors first and building the noise generator from them. It still turned out to be the main limitation (section 7).

## 3. The pipeline, step by step

```
Phase 1  Real pages ──► Tesseract ──► compare with truth ──► which errors does real OCR make?   (baseline CER 9.6%)
                                                              │
Phase 2  Sanskrit Wikipedia ──► 436k clean lines ──┬──► rule-based noise (copies those errors)  ─┐
                                                   └──► render as images ──► Tesseract (real errors)┴──► 68k training pairs
Phase 3  ByT5-small ──► fine-tune on (noisy → clean) pairs ──► best checkpoint by dev CER
Phase 4  Evaluate on: real pages · rendered images · synthetic noise ──► CER, WER, over-correction, error types, CIs
Phase 5  Publish to Hugging Face ──► reproduce from a fresh clone
```

### Phase 1: measure the real problem first
**What:** I ran Tesseract (Sanskrit model) on the 5 given pages and compared its output with the true text.
**Why first:** I didn't want to *guess* what OCR errors look like. I counted them:
- about 60% were look-alike letter swaps (य→म 8 times, व→ठ, ह→द)
- about 25% were missing or wrong marks (`ं`→`े`, a dropped `ः`)
- about 10% were broken conjuncts
- a few were **real-word errors**: OCR produced a *different valid word* (रोगः "disease" → योगः "yoga"). No dictionary can catch these.

**Result:** baseline **CER 9.6%, WER 33.9%**. One wrong letter ruins a whole word, which is why WER is so much higher.

### Phase 2: build training data without labelling anything
- **Clean text:** Sanskrit Wikipedia, cut into line-length pieces (3–10 words, like the test pages). That gave 436k unique lines. I removed anything overlapping the test pages, so the model never sees them.
- **Noise source 1, rules:** a generator (`src/noise.py`) that makes the same kinds of mistakes in the same proportions as real Tesseract. **Check:** its error mix (69% substitutions, 17% deletions, 13% insertions) closely matches real OCR (75/16/10).
- **Noise source 2, real OCR on rendered images:** I drew 6,000 lines as degraded "scans" (6 fonts, blur, noise, rotation, compression) and ran Tesseract on them. Those errors are *real* OCR errors.
  - Tesseract read clean print fonts almost perfectly (0–4% CER), which teaches little. It struggled on a handwriting-style font (Kalam) that resembles the test pages, so I weighted renders toward it. I report this openly because it may flatter the results.
- **24% of training inputs are already correct**, on purpose, so the model learns to *leave correct text alone*.

### Phase 3: choose and train the model
See section 4 for why ByT5. Training settings and the reason for each:

| Setting | Why |
|---|---|
| Full fine-tuning, not LoRA | 300M parameters fit on a T4, and LoRA adds complexity with no memory need here |
| fp32 | The T4 has no bf16, and T5-family models produce NaN losses in fp16 |
| Batch 8 × 4 gradient accumulation | Batch 16 ran out of memory in fp32 on the longest lines |
| Adafactor optimizer | T5's native optimizer, with much less memory than AdamW |
| Length-grouped batches | Bytes make sequences long, so grouping similar lengths avoids wasted padding |
| Checkpoints to Google Drive every 400 steps, resumable | Colab runtimes reset, so no run should be lost |
| Best checkpoint chosen by **dev CER on generated text**, not loss | Loss can fall while real corrections don't improve |

**Why training stopped early:** the Colab GPU kept disconnecting. I stopped at step 1,768 of 2,125 and used the best checkpoint, `checkpoint-1600` (dev CER 5.2%, still improving).

### Phase 4: evaluate honestly
I tested on three sets the model never trained on:
1. **The 5 real pages.** This is the result that matters.
2. **500 rendered-image lines** read by real Tesseract.
3. **1,000 synthetic-noise lines.**

Beyond CER and WER, I measured:
- **over-correction:** how often it *breaks* a line that was already correct, which is this task's version of hallucination
- the share of lines fully fixed and the share made worse
- errors by type
- results by font
- greedy vs beam search
- **95% confidence intervals**, because 5 pages is a tiny sample

### Phase 5: make it reproducible
The model is public on Hugging Face. A fresh clone plus the published model reproduces the real-page result exactly with one command (`src/evaluate.py`).

## 4. Why ByT5-small (and not the other models)

| Candidate | Why not, or why |
|---|---|
| **ByT5-small** ✅ | It reads and writes **raw bytes**, so it has no vocabulary that can fail. OCR output contains junk characters (`|`, `l`, `:`), broken vowel-sign sequences and rare conjuncts, and a byte model handles any of them. A correction is a tiny byte edit. It's 300M parameters, so it can be fully fine-tuned on a T4. |
| mT5-small | Same size and architecture, with a subword vocabulary. **I tested it rather than assuming:** it handled Devanagari *better than I expected* (0% unknown tokens, 99.9% exact round-trip). So I didn't choose ByT5 for a tokenizer failure. I chose it because it's lossless by design and published results show byte models are more robust on noisy text. mT5 is about 6× faster (shorter sequences), so it's the first thing I'd compare next. |
| Llama-3.2-1B / Gemma-2B (LLMs) | 3–7× larger for a narrow copy-and-edit task, needs QLoRA on a T4, and decoder-only chat models paraphrase more readily, which means more over-correction. |
| IndicTrans2 | Built for *translation* between languages, not same-language correction. |
| A spell-checker / dictionary | Can't fix real-word errors (योगः is a valid word), and Sanskrit's compounding and sandhi make dictionary coverage poor. |

**The trade-off I accepted:** bytes make sequences about 6.3× longer than subword tokens, so training and generation are slower. That's the main reason I could train only one epoch.

## 5. Results, and what they mean

| Test set | CER before → after (95% CI) |
|---|---|
| **Real page images** | 9.6% [7.4–12.5] → **8.2%** [6.4–10.9]; beam 4: **7.7%** |
| Rendered images (real Tesseract) | 8.8% [8.0–9.7] → **7.4%** [6.7–8.2] |
| Synthetic noise | 7.4% [7.0–7.8] → **3.4%** [3.2–3.7] |

**What it means:**
- On noise like its training data it **more than halves** the errors, with a clear, statistically solid gain.
- On real Tesseract output it cuts errors by **15–20%**. On only 5 pages the confidence intervals overlap, so I can't claim that's statistically proven.
- It fixes letter swaps and broken conjuncts well (−64% and −58% on synthetic noise) but barely fixes marks (−11%), because often both forms are valid.
- It helps where OCR is bad (handwriting-style font: 15.2% → 12.2%) and slightly hurts where OCR is already good (clean fonts at about 2%). **In a real system I'd only send low-confidence OCR lines to it.**

## 6. What went wrong, and why (failure analysis)
- **Over-correction: 11% of already-correct lines were changed.** The model learned my look-alike pairs in both directions (घ↔ध, क↔फ), so on rare words it "fixes" correct letters. On real page 2 it turned the Ayurvedic term कफः into ककः twice. Cause: the two-way confusion table, plus कफ being rare in Wikipedia.
- **Real-word errors: 0 of 3 fixed.** योगः, द्रोण and निहार are valid words, and fixing them needs domain knowledge (Ayurveda) that Wikipedia training doesn't give.
- **Adding marks too eagerly:** रोग→रोगं, बल→बलं. My noise drops anusvara often, so the model learned to add it back too often.
- **Digits can't be inferred** from context (२०११ → २०१९ stays wrong).

## 7. Biggest lesson
**The gap between synthetic and real errors decides everything.** The model learned *my* noise model very well (−54%), but real OCR is only partly covered by it (−15–20%). The best next investments aren't a bigger model but better data:
- one-way, data-driven confusions
- classical and Ayurvedic text instead of Wikipedia
- more real scanned pages for testing
- gating corrections on the model's confidence, to stop over-correction

## 8. Engineering problems I hit and solved
- **Library changes:** Colab ships `transformers` 5.x, which renamed training arguments. I inspected the signatures and fixed both at once.
- **Out of GPU memory on step 1:** length-grouping deliberately puts the longest batch first, so I fixed it with gradient accumulation at the same effective batch size.
- **Notebook freezing:** a background launch with `!nohup … &` stayed attached to the cell. I switched to a properly detached process.
- **Colab GPU disconnects:** resumable training with checkpoints on Drive meant no lost work. In the end I stopped early and used the best checkpoint.
- **A bug in my own evaluation:** a substring check wrongly said योगः was unfixed inside the correct word संयोगः. I switched to whole-word comparison.

## 9. What I'd do with more time
1. Finish the epoch and train 2–3 epochs. The dev curve was still falling.
2. Use one-way confusions and a lower anusvara-drop rate, the two causes of over-correction I measured.
3. Gate corrections on confidence: apply an edit only when the model is sure.
4. Use better text (GRETIL, DCS, Ayurvedic texts) and more real scanned test pages.
5. Compare against mT5-small, which is about 6× faster.
6. Deploy as an int8/ONNX model for CPU behind a simple OCR → correct → review screen.

---

## Likely interview questions, with short answers
**Why Option 3?** It's the only track where I could measure progress objectively on real data within the time and the T4 budget. There's exact ground truth, free training data and a clear "do nothing" baseline.

**Why ByT5 if mT5's tokenizer handled Sanskrit fine?** I tested that and was surprised. I kept ByT5 because it's lossless by design for arbitrary OCR junk and reported to be more robust on noisy text. The cost is 6× longer sequences, so mT5 is my first planned comparison.

**Why not LoRA?** 300M parameters fit fully on a T4. LoRA solves a memory problem I didn't have.

**How do you know there's no data leakage?** Clean lines are split so no line is in two splits, and any line sharing a 3-word window with the test pages was removed. Both are checked with assertions in the notebook. The real pages are never used for training; only their error *proportions* informed the noise model, and I state that.

**Your real-page gain isn't significant, so is the model useful?** On 705 characters, no single result can be significant. The gains on 500 rendered lines read by real Tesseract are significant, and the failure analysis shows exactly what to fix. I'd rather report it honestly than overclaim.

**What's the biggest risk in production?** Over-correction: silently changing correct text. That's why I measure it, and why I'd gate edits by confidence and only send low-confidence OCR lines to the model.

**Why did training stop early?** Colab's GPU runtime kept disconnecting. Checkpoints on Drive meant nothing was lost. I used the best checkpoint (dev CER 5.2%) and documented that the model is partly trained.
