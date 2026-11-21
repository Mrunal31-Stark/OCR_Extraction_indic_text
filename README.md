# Post-OCR Correction for Sanskrit (Devanagari)

Fine-tunes **ByT5-small** to clean up Tesseract output on Sanskrit text: a noisy OCR line goes in and the corrected line comes out.
This is Option 3 of the AI/ML Model Training assignment. The full write-up is in [`REPORT.md`](REPORT.md).

```
Tesseract (san)   →  "आयुः तस्त्र हितं च अदितं च मानम् च तच् च मत्र उक्तं"
ByT5 corrector    →  "आयुः तस्य हितं च अदितं च मानम् च तच् च यत्र उक्तं"
Ground truth      →  "आयुः तस्य हितं च अहितं च मानम् च तच् च यत्र उक्तं"
```
(Real page 1: two errors fixed, one real-word-style error, अदितं, missed.)

## Repository layout
| Path | What it is |
|---|---|
| `notebooks/Sanskrit_Post_OCR_Correction.ipynb` | End-to-end Colab notebook (Phases 0–4), every code cell with its output |
| `src/noise.py` | Synthetic OCR-noise generator, calibrated on real Tesseract errors |
| `src/train.py` | ByT5 fine-tuning (HF `Seq2SeqTrainer`, resumable, checkpoints to Drive) |
| `src/infer.py` | Inference CLI / `Corrector` class (batched, splits long lines) |
| `src/evaluate.py` | Metrics: CER/WER, over-correction, error-type breakdown, bootstrap CIs |
| `Data/` | **Given real test set:** 5 page images + ground-truth text (never trained on) |
| `datasets/` | All generated data: `train.jsonl` (67,984 pairs), `dev.jsonl` (1,500), `test_rule.jsonl` (1,000), `test_render.jsonl` (500), `render_pairs.jsonl` (6,000 rendered-image OCR pairs, by font), `phase1.json` (Tesseract output + error analysis of the real pages) |
| `results/metrics.json` | **All metrics:** CER/WER before→after with **95% bootstrap CIs**, over-correction / fully-fixed / made-worse rates and counts, error types, per-page and per-font results, beam search, dev learning curve |
| `results/predictions_*.jsonl` | Model output for **every test line** with per-line CER and flags (`over_corrected`, `made_worse`, `fully_fixed`) |
| `results/predictions_real_pages.json` | Real pages: ground truth / Tesseract / corrected, per page and per line |
| `results/trainer_state_checkpoint-1600.json`, `results/train_resumed_from_400.log` | Training log: loss and dev CER/WER per evaluation step |
| `samples/` | 6 **noisy images we created** from held-out text + their OCR and corrected outputs (`samples.json`) |
| `REPORT.md` | Report: the 9 required points, results, failure analysis, limitations |

Each JSONL line in `datasets/` is `{"clean": …, "noisy": …, "src": "rule" | "render", "font": … (render only)}`.

## Setup (Google Colab, T4)
1. Open the notebook in Colab (File → Open notebook → GitHub → this repo) and choose **Runtime → Change runtime type → T4 GPU**.
2. Run the cells top to bottom. The notebook clones this repo for the test data, installs Tesseract (`san`) and the Python dependencies, and mounts Google Drive for checkpoints.
3. **Phase 2 takes ~20 min:** rendering 6k lines and running OCR on them, on 2 CPUs. To skip it, use the ready-made splits in `datasets/`.
4. **Training takes ~1.5 h on a T4** (fp32). It resumes from the latest Drive checkpoint if the runtime resets.

Local setup (CPU inference only):
```bash
pip install torch transformers datasets jiwer
python src/infer.py --model path/to/checkpoint-1600 --text "योगः नाम दोष वेषम्म कारणम्"
python src/noise.py      # self-check
python src/evaluate.py   # self-check
```

## Key results
Model: `checkpoint-1600` (best dev CER, 5.2%). Numbers come from [`results/metrics.json`](results/metrics.json); the full analysis is in [`REPORT.md`](REPORT.md#6-evaluation-methodology).

| Test set | CER before → after (95% CI) | WER before → after | Over-correction |
|---|---|---|---|
| **Given real page images** (5 pages) | 9.6% [7.4–12.5] → **8.2%** [6.4–10.9]; beam 4: **7.7%** | 33.9% → 32.2% | – |
| **Our noisy rendered images** (500 lines) | 8.8% [8.0–9.7] → **7.4%** [6.7–8.2] | 32.6% → 27.8% | 11 / 155 (7.1%) |
| Synthetic rule noise (1,000 lines) | 7.4% [7.0–7.8] → **3.4%** [3.2–3.7] | 36.0% → 18.5% | 32 / 233 (13.7%) |

Statistically clear gains on both generated test sets (non-overlapping CIs). On the 5 real pages the gain is modest and the CIs overlap. 11% of already-correct lines get changed (over-correction). See the report for the failure analysis.

## Model checkpoint
**Hugging Face:** [`MrunalYewatkar45/byt5-small-sanskrit-post-ocr`](https://huggingface.co/MrunalYewatkar45/byt5-small-sanskrit-post-ocr) (public, `checkpoint-1600`, 1.2 GB, with model card).

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
repo = "MrunalYewatkar45/byt5-small-sanskrit-post-ocr"
tok, model = AutoTokenizer.from_pretrained(repo), AutoModelForSeq2SeqLM.from_pretrained(repo)
x = tok(["आयुः तस्त्र हितं च अदितं च मानम् च"], return_tensors="pt")
print(tok.batch_decode(model.generate(**x, max_length=384), skip_special_tokens=True))
# -> ['आयुः तस्य हितं च अदितं च मानम् च']
```
Or with the repo's CLI: `python src/infer.py --model MrunalYewatkar45/byt5-small-sanskrit-post-ocr --text "…"`.

The weights aren't committed to GitHub because its file limit is 100 MB.
