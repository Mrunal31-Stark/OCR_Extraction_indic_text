# Post-OCR Correction for Sanskrit (Devanagari)

Fine-tunes **ByT5-small** to clean up Tesseract output on Sanskrit text: a noisy OCR line goes in and the corrected line comes out.
This is Option 3 of the AI/ML Model Training assignment. The full write-up is in [`REPORT.md`](REPORT.md).

```
OCR (Tesseract san)  →  "तस्त्र हितं च अदितं च"
ByT5 corrector       →  "तस्य हितं च अहितं च"
```

## Repository layout
| Path | What it is |
|---|---|
| `notebooks/Sanskrit_Post_OCR_Correction.ipynb` | End-to-end Colab notebook (Phases 0–4) with outputs |
| `src/noise.py` | Synthetic OCR-noise generator, calibrated on real Tesseract errors |
| `src/train.py` | ByT5 fine-tuning (HF `Seq2SeqTrainer`, resumable, checkpoints to Drive) |
| `src/infer.py` | Inference CLI / `Corrector` class (batched, splits long lines) |
| `src/evaluate.py` | Metrics: CER/WER, over-correction, error-type breakdown, bootstrap CIs |
| `Data/` | Held-out **real** test set: 5 page images + ground-truth text (never trained on) |
| `REPORT.md` | Report covering the 9 required points, with limitations |

## Setup (Google Colab, T4)
1. Open the notebook in Colab and choose **Runtime → Change runtime type → T4 GPU**.
2. Run the cells top to bottom. The notebook clones this repo for the test data, installs Tesseract (`san`) and the Python dependencies, and mounts Google Drive for checkpoints.
3. **Phase 2 takes ~20 min:** rendering 6k lines and running OCR on them, on 2 CPUs. Its outputs are backed up to `MyDrive/sanskrit_ocr/`.
4. **Training takes ~1–1.5 h on a T4** (fp32). It resumes from the latest Drive checkpoint if the runtime resets.

Local setup (CPU inference only):
```bash
pip install torch transformers datasets jiwer
python src/infer.py --model path/to/final --text "योगः नाम दोष वेषम्म कारणम्"
python src/noise.py      # self-check
python src/evaluate.py   # self-check
```

## Key results
Results are filled in after Phase 4. See [`REPORT.md`](REPORT.md#6-evaluation-methodology).

| Test set | CER before | CER after | WER before | WER after |
|---|---|---|---|---|
| Real pages (5) | 9.6% | – | 33.9% | – |
| Synthetic (rule noise) | 7.4% | – | – | – |
| Rendered + OCR | 8.8% | – | – | – |

## Model checkpoint
Trained weights (~1.2 GB) are on Google Drive at `MyDrive/sanskrit_ocr/byt5/final`. They aren't committed here because of their size.
