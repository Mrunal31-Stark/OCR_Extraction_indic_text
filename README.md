# Post-OCR Correction for Sanskrit (Devanagari)

Fine-tunes **ByT5-small** to clean up Tesseract output on Sanskrit text: a noisy OCR line goes in and the corrected line comes out.
This is Option 3 of the AI/ML Model Training assignment. The full write-up is in [`REPORT.md`](REPORT.md).

## Links
| | |
|---|---|
| **Trained model (Hugging Face, public)** | https://huggingface.co/MrunalYewatkar45/byt5-small-sanskrit-post-ocr |
| **Code, data, results (this repo)** | https://github.com/Mrunal31-Stark/OCR_Extraction_indic_text |
| **Report** | [`REPORT.md`](REPORT.md) |
| **Notebook** | [`notebooks/Sanskrit_Post_OCR_Correction.ipynb`](notebooks/Sanskrit_Post_OCR_Correction.ipynb) |

```
Tesseract (san)   →  "आयुः तस्त्र हितं च अदितं च मानम् च तच् च मत्र उक्तं"
ByT5 corrector    →  "आयुः तस्य हितं च अदितं च मानम् च तच् च यत्र उक्तं"
Ground truth      →  "आयुः तस्य हितं च अहितं च मानम् च तच् च यत्र उक्तं"
```
(Real page 1: two errors fixed, one real-word-style error, अदितं, missed.)

> **Training status:** training was **stopped early, at step 1,768 of 2,125 (83% of one epoch)**, because the Google Colab GPU runtime kept disconnecting. The released model is the best saved checkpoint, **`checkpoint-1600`** (dev CER 5.2%, still improving). Details are in [`REPORT.md` §4](REPORT.md#4-fine-tuning-approach).

## Key results
Numbers come from [`results/metrics.json`](results/metrics.json); the full analysis is in [`REPORT.md`](REPORT.md#6-evaluation-methodology).

| Test set | CER before → after (95% CI) | WER before → after | Over-correction |
|---|---|---|---|
| **Given real page images** (5 pages) | 9.6% [7.4–12.5] → **8.2%** [6.4–10.9]; beam 4: **7.7%** | 33.9% → 32.2% | – |
| **My noisy rendered images** (500 lines) | 8.8% [8.0–9.7] → **7.4%** [6.7–8.2] | 32.6% → 27.8% | 11 / 155 (7.1%) |
| Synthetic rule noise (1,000 lines) | 7.4% [7.0–7.8] → **3.4%** [3.2–3.7] | 36.0% → 18.5% | 32 / 233 (13.7%) |

Statistically clear gains on both generated test sets (non-overlapping CIs). On the 5 real pages the gain is modest and the CIs overlap. 11% of already-correct lines get changed (over-correction). See the report for the failure analysis.

## Repository layout
| Path | What it is |
|---|---|
| `notebooks/Sanskrit_Post_OCR_Correction.ipynb` | End-to-end Colab notebook (Phases 0–5: baseline → data → training → evaluation → Hugging Face publish + reproduction check), every code cell with its output (except the interactive login cell) |
| `src/noise.py` | Synthetic OCR-noise generator, calibrated on real Tesseract errors |
| `src/train.py` | ByT5 fine-tuning (HF `Seq2SeqTrainer`, resumable from the latest checkpoint) |
| `src/infer.py` | Inference CLI / `Corrector` class (batched, splits long lines) |
| `src/evaluate.py` | Evaluation CLI + metrics: CER/WER, over-correction, error types, bootstrap CIs |
| `requirements.txt` | Python dependencies |
| `Data/` | **Given real test set:** 5 page images + ground-truth text (never trained on) |
| `datasets/` | All generated data: `train.jsonl` (67,984 pairs), `dev.jsonl` (1,500), `test_rule.jsonl` (1,000), `test_render.jsonl` (500), `render_pairs.jsonl` (6,000 rendered-image OCR pairs, by font), `phase1.json` (saved Tesseract output + error analysis of the real pages) |
| `results/metrics.json` | **All metrics:** CER/WER before→after with **95% bootstrap CIs**, over-correction / fully-fixed / made-worse rates and counts, error types, per-page and per-font results, beam search, dev learning curve |
| `results/predictions_*.jsonl` | Model output for **every test line** with per-line CER and flags (`over_corrected`, `made_worse`, `fully_fixed`) |
| `results/predictions_real_pages.json` | Real pages: ground truth / Tesseract / corrected, per page and per line |
| `results/trainer_state_checkpoint-1600.json`, `results/train_resumed_from_400.log` | Training log: loss and dev CER/WER per evaluation step |
| `samples/` | 6 **noisy images I created** from held-out text + their OCR and corrected outputs (`samples.json`) |
| `REPORT.md` | Report: the 9 required points, results, failure analysis, limitations |

Each JSONL line in `datasets/` is `{"clean": …, "noisy": …, "src": "rule" | "render", "font": … (render only)}`.

---

## Setup

There are three ways to use this project, from quickest to most complete:

| Goal | Where | Time | GPU? |
|---|---|---|---|
| **A.** Run the model / reproduce the evaluation | Colab or local | 5–15 min | No (faster with one) |
| **B.** OCR + correct your own page images | Local or Colab | 10 min setup | No |
| **C.** Reproduce everything (data → training → evaluation) | Colab (T4) | ~2.5 h | Yes |

### Option A1: Google Colab, run the model and reproduce the evaluation
1. Go to https://colab.research.google.com → **New notebook**.
2. *(Optional, faster)* **Runtime → Change runtime type → T4 GPU → Save**. CPU works too, but evaluation is about 10× slower.
3. Paste this into a cell and run it:
   ```python
   !git clone -q https://github.com/Mrunal31-Stark/OCR_Extraction_indic_text.git
   %cd OCR_Extraction_indic_text
   !pip -q install -r requirements.txt
   ```
4. Correct a line (the model, about 1.2 GB, downloads from Hugging Face on first use):
   ```python
   !python src/infer.py --model MrunalYewatkar45/byt5-small-sanskrit-post-ocr --text "योगः नाम दोष वेषम्म कारणम्"
   # -> योगः नाम दोष वेषम्य कारणम्
   ```
5. Reproduce the reported numbers. `--real-pages` uses the Tesseract output saved in `datasets/phase1.json`, so Tesseract isn't needed:
   ```python
   !python src/evaluate.py --model MrunalYewatkar45/byt5-small-sanskrit-post-ocr \
        --real-pages datasets/phase1.json \
        --data datasets/test_render.jsonl datasets/test_rule.jsonl \
        --out my_metrics.json
   ```
   Expected output on the real pages: `CER 0.0965 -> 0.0823 | WER 0.3391 -> 0.3217`. This was checked on a fresh clone. The two test sets take about 3 min on a T4; add `--limit 100` for a quick run on CPU. Add `--beams 4` for beam search (real pages → 7.7%).

### Option A2: Local machine, run the model and reproduce the evaluation
**Requirements:** Python **3.10–3.13**, about 3 GB free disk (model + PyTorch), and internet access to `huggingface.co` the first time. An NVIDIA GPU is optional; CPU works.

1. **Get the code**
   ```bash
   git clone https://github.com/Mrunal31-Stark/OCR_Extraction_indic_text.git
   cd OCR_Extraction_indic_text
   ```
2. **Create a virtual environment**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install PyTorch.** Pick **one**:
   ```bash
   pip install torch                                                        # CPU only (any OS)
   pip install torch --index-url https://download.pytorch.org/whl/cu121     # NVIDIA GPU (CUDA 12.x)
   ```
4. **Install the rest**
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the self-checks** (no model needed, a few seconds):
   ```bash
   python src/noise.py       # prints "ok [...]"
   python src/evaluate.py    # prints "ok {...}"
   ```
6. **Correct text**
   ```bash
   python src/infer.py --model MrunalYewatkar45/byt5-small-sanskrit-post-ocr --text "आयुः तस्त्र हितं च अदितं च"
   python src/infer.py --model MrunalYewatkar45/byt5-small-sanskrit-post-ocr --file my_ocr_output.txt   # one line per line
   ```
7. **Reproduce the evaluation:** same command as Option A1, step 5. On CPU expect roughly 0.5–2 s per line, so use `--limit 100` for a quick check. The real pages alone take under a minute.

*Note for Windows users:* if Devanagari shows up as `????` in the terminal, run `chcp 65001` (or set `PYTHONIOENCODING=utf-8`) first. If `huggingface.co` is blocked on your network, download the model files from the Hugging Face page in a browser and pass the local folder to `--model`.

### Option B: OCR and correct your own page images (local or Colab)
This needs the **Tesseract** program with the Sanskrit (`san`) language data:
- **Colab / Ubuntu / Debian:** `sudo apt-get install -y tesseract-ocr tesseract-ocr-san` (in Colab, prefix with `!` and drop `sudo`)
- **macOS:** `brew install tesseract tesseract-lang`
- **Windows:**
  1. Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki (the default install path is `C:\Program Files\Tesseract-OCR`).
  2. During installation, tick **Additional language data → Sanskrit**. Alternatively, download `san.traineddata` from https://github.com/tesseract-ocr/tessdata and copy it into `C:\Program Files\Tesseract-OCR\tessdata`.
  3. Check it with: `tesseract --list-langs` (the list must include `san`).

Then:
```python
import sys; sys.path.insert(0, 'src')
import pytesseract
from PIL import Image
from infer import Corrector
# Windows only, if tesseract isn't on PATH:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ocr = pytesseract.image_to_string(Image.open('Data/page_1.png'), lang='san', config='--psm 6')   # same config as my baseline
fixed = Corrector('MrunalYewatkar45/byt5-small-sanskrit-post-ocr').correct(ocr.splitlines(), num_beams=4)
print('\n'.join(fixed))
```
Tip: the corrector helps most on low-quality OCR. On clean, already-accurate OCR it can over-correct (see the report).

### Option C: Reproduce everything in Google Colab (data → training → evaluation)
1. **Open the notebook:** in Colab, **File → Open notebook → GitHub tab**, paste `Mrunal31-Stark/OCR_Extraction_indic_text`, and pick `notebooks/Sanskrit_Post_OCR_Correction.ipynb`.
2. **Select a GPU:** **Runtime → Change runtime type → T4 GPU → Save**. An L4 or A100 is faster; the script switches to bf16 on them automatically.
3. **Run Phase 0 (setup, ~2 min):** the first code cell installs Tesseract (`san`, `hin`) and `jiwer`. The next cell mounts **Google Drive**: click **Connect to Google Drive → Allow**. Drive holds the checkpoints, so training survives runtime resets.
4. **Run Phase 1 (OCR baseline, ~1 min):** clones this repo for `Data/`, runs Tesseract on the 5 pages, prints CER/WER and the error analysis, and runs the sanity checks.
5. **Run Phase 2 (training data, ~25 min):** downloads Sanskrit Wikipedia, builds the noise model, renders and OCRs 6,000 line images (the render cell runs in batches of 1,500, so **re-run it until it prints `rendered 6000/6000`**), then writes the splits and backs them up to `MyDrive/sanskrit_ocr/`.
   *Shortcut: skip Phase 2* by copying the ready-made splits from this repo to Drive and to the working folder, then continue with Phase 3:
   ```python
   !mkdir -p /content/drive/MyDrive/sanskrit_ocr /content/work
   !cp /content/OCR_Extraction_indic_text/datasets/* /content/drive/MyDrive/sanskrit_ocr/
   !cp /content/OCR_Extraction_indic_text/datasets/* /content/work/
   ```
6. **Run Phase 3 (training, ~1.5 h on a T4):**
   - The tokenizer cell and the smoke-run cell (40 steps, about 4 min) are optional checks.
   - The **"Resume training from the latest Drive checkpoint"** cell starts training as a **detached background process**: the cell returns after about 2 minutes and the notebook stays usable.
   - Re-run the **monitor** cell any time to see the step, loss and dev CER (evaluated every 400 steps, checkpoints saved to `MyDrive/sanskrit_ocr/byt5/`).
   - **If Colab disconnects:** reconnect and re-run the resume cell. It restores the data from Drive and continues from the last saved checkpoint, losing at most 400 steps.
   - Keep the browser tab open; Colab may stop idle or long-running sessions.
   - **Do not run the "3d. Early stop" cell** unless you want to stop training. It kills the training process. I used it because of repeated disconnects; if training finishes normally, the final model is saved to `MyDrive/sanskrit_ocr/byt5/final`.
7. **Run Phase 4 (evaluation, ~10 min):** loads the best checkpoint from Drive and evaluates the real pages, the rendered-image test set (by font) and the rule-noise test set. It also renders the 6 sample images and runs the failure analysis. The last cell packages all results and downloads them as a zip.

**Colab troubleshooting**
| Problem | Fix |
|---|---|
| `nvidia-smi: command not found` / `No GPU` | Runtime → Change runtime type → T4 GPU (this restarts the runtime: re-run from Phase 0) |
| `CUDA out of memory` at the first step | Keep the default `--bs 8 --accum 4`. Batches are length-grouped, so the longest batch comes first. |
| `NaN` loss | Don't use fp16 with T5/ByT5. The script uses fp32 on a T4 and bf16 on L4/A100 automatically. |
| Runtime disconnected mid-training | Reconnect → re-run the resume cell (continues from the last Drive checkpoint) |
| `TypeError: … unexpected keyword argument 'warmup_ratio'` | You have transformers 5.x with an old script. Use this repo's `src/train.py`, which uses the 5.x argument names. |
| Tesseract `san` missing | `!apt-get -qq install -y tesseract-ocr-san` |

### Training locally (optional)
Needs an NVIDIA GPU with **at least 12 GB** of memory for the default settings (fp32, batch 8 × accumulation 4, 384 bytes). Use `--bs 4 --accum 8` for about 8 GB.
```bash
python src/train.py --data datasets --out checkpoints/byt5 --eval_steps 400            # 1 epoch, resumes automatically
python src/evaluate.py --model checkpoints/byt5/checkpoint-XXXX --real-pages datasets/phase1.json --data datasets/test_render.jsonl
```

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
The weights aren't committed to GitHub because its file limit is 100 MB.
