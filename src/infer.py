"""Correct OCR output with the fine-tuned ByT5 model.

Usage:
  python infer.py --model path/to/final --text "तस्त्र हितं च अदितं च"
  python infer.py --model path/to/final --file ocr_output.txt      # one OCR line per line
"""
import argparse
import unicodedata

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_BYTES = 384


class Corrector:
    def __init__(self, model_dir, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device).eval()

    @staticmethod
    def _split(line, max_bytes=300):
        """Lines longer than the training length are corrected in word-aligned pieces."""
        pieces, cur = [], ''
        for w in line.split():
            if cur and len((cur + ' ' + w).encode()) > max_bytes:
                pieces.append(cur); cur = w
            else:
                cur = f'{cur} {w}'.strip()
        return pieces + ([cur] if cur else [])

    @torch.no_grad()
    def correct(self, lines, batch_size=32, num_beams=1):
        lines = [unicodedata.normalize('NFC', l.strip()) for l in lines]
        pieces, owner = [], []
        for i, l in enumerate(lines):
            for p in self._split(l) or ['']:
                pieces.append(p); owner.append(i)
        outs = []
        for s in range(0, len(pieces), batch_size):
            batch = pieces[s:s + batch_size]
            enc = self.tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=MAX_BYTES).to(self.device)
            gen = self.model.generate(**enc, max_length=MAX_BYTES, num_beams=num_beams)
            outs += self.tok.batch_decode(gen, skip_special_tokens=True)
        result = [''] * len(lines)
        for i, o in zip(owner, outs):
            result[i] = f'{result[i]} {o}'.strip()
        return [unicodedata.normalize('NFC', r) for r in result]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--text')
    ap.add_argument('--file')
    ap.add_argument('--beams', type=int, default=1)
    a = ap.parse_args()
    lines = [a.text] if a.text else open(a.file, encoding='utf-8').read().splitlines()
    for line in Corrector(a.model).correct(lines, num_beams=a.beams):
        print(line)
