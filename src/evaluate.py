"""Evaluation for post-OCR correction: CER/WER, over-correction, per-error-type breakdown, bootstrap CIs.

All metrics compare NFC-normalized strings. Used by the notebook (Phase 4); `python evaluate.py` runs a self-check.
"""
import difflib
import random
from collections import Counter

import jiwer

MARKS = set('ािीुूृॄेैोौंःँ़')
VIRAMA = '्'


def edits(ref, hyp):
    """Code-point level edit ops (tag, ref_span, hyp_span) turning ref into hyp."""
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return [(t, ref[i1:i2], hyp[j1:j2]) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != 'equal']


def category(src, dst):
    s = src + dst
    if any(c in ' ।॥|:.,l' for c in s):
        return 'space/punct'
    if VIRAMA in s:
        return 'conjunct'
    if any(c in MARKS for c in s):
        return 'mark'
    return 'consonant'


def error_types(refs, hyps):
    return Counter(category(s, d) for r, h in zip(refs, hyps) for _, s, d in edits(r, h))


def line_report(clean, noisy, pred):
    """Line-level behaviour of the corrector."""
    n = len(clean)
    already_ok = [i for i in range(n) if noisy[i] == clean[i]]
    broken = [i for i in range(n) if noisy[i] != clean[i]]
    cer = lambda r, h: jiwer.cer(r, h) if r else float(h != r)
    worse = sum(cer(clean[i], pred[i]) > cer(clean[i], noisy[i]) for i in range(n))
    return {
        'lines': n,
        'fully_fixed': sum(pred[i] == clean[i] for i in broken) / max(1, len(broken)),
        'over_correction': sum(pred[i] != clean[i] for i in already_ok) / max(1, len(already_ok)),
        'made_worse': worse / n,
    }


def bootstrap_cer(refs, hyps, n=1000, seed=0):
    """95% CI for corpus CER by resampling lines (important for the tiny real test set)."""
    rng, idx, vals = random.Random(seed), range(len(refs)), []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(jiwer.cer([refs[i] for i in s], [hyps[i] for i in s]))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def full_report(clean, noisy, pred):
    return {
        'cer_before': jiwer.cer(clean, noisy), 'cer_after': jiwer.cer(clean, pred),
        'wer_before': jiwer.wer(clean, noisy), 'wer_after': jiwer.wer(clean, pred),
        **line_report(clean, noisy, pred),
        'types_before': dict(error_types(clean, noisy)), 'types_after': dict(error_types(clean, pred)),
    }


def flat(s):
    return ' '.join(s.split())


def main(argv):
    """CLI: evaluate a model on a jsonl test set and/or the real pages (uses the Tesseract output saved in phase1.json)."""
    import argparse
    import json
    from infer import Corrector
    ap = argparse.ArgumentParser(description='Evaluate a post-OCR corrector.')
    ap.add_argument('--model', required=True, help='HF repo id or local checkpoint dir')
    ap.add_argument('--data', nargs='*', default=[], help='jsonl files with {"clean", "noisy"} per line')
    ap.add_argument('--real-pages', help='datasets/phase1.json (ground truth + saved Tesseract output)')
    ap.add_argument('--limit', type=int, help='evaluate only the first N lines of each jsonl (quick check on CPU)')
    ap.add_argument('--beams', type=int, default=1)
    ap.add_argument('--out', help='write all metrics to this json file')
    a = ap.parse_args(argv)
    corr, out = Corrector(a.model), {}
    for path in a.data:
        rows = [json.loads(l) for l in open(path, encoding='utf-8')][:a.limit]
        clean, noisy = [r['clean'] for r in rows], [r['noisy'] for r in rows]
        pred = corr.correct(noisy, num_beams=a.beams)
        rep = full_report(clean, noisy, pred)
        rep['cer_before_ci95'], rep['cer_after_ci95'] = bootstrap_cer(clean, noisy, n=300), bootstrap_cer(clean, pred, n=300)
        out[path] = rep
        print(f"{path}: n={len(rows)} CER {rep['cer_before']:.4f} -> {rep['cer_after']:.4f} | WER {rep['wer_before']:.4f} -> "
              f"{rep['wer_after']:.4f} | fully fixed {rep['fully_fixed']:.1%} | over-correction {rep['over_correction']:.1%} | "
              f"made worse {rep['made_worse']:.1%}\n  error types {rep['types_before']} -> {rep['types_after']}")
    if a.real_pages:
        p1 = json.load(open(a.real_pages, encoding='utf-8'))
        gt, ocr = p1['gt'], p1['ocr'][p1['best_config']]
        G, O, F = [], [], []
        for k in sorted(gt, key=int):
            fixed = corr.correct(ocr[k].splitlines(), num_beams=a.beams)
            G.append(flat(gt[k])); O.append(flat(ocr[k])); F.append(flat(' '.join(fixed)))
            print(f"page {k}: CER {jiwer.cer(G[-1], O[-1]):.4f} -> {jiwer.cer(G[-1], F[-1]):.4f}\n  FIXED: {F[-1]}")
        rep = full_report(G, O, F)
        out['real_pages'] = rep
        print(f"real pages: CER {rep['cer_before']:.4f} -> {rep['cer_after']:.4f} | WER {rep['wer_before']:.4f} -> {rep['wer_after']:.4f}")
    if a.out:
        json.dump(out, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__' and len(__import__('sys').argv) > 1:
    main(__import__('sys').argv[1:])
elif __name__ == '__main__':
    assert category('य', 'म') == 'consonant' and category('ं', 'े') == 'mark'
    assert category('', '्र') == 'conjunct' and category(' ', '') == 'space/punct'
    clean = ['रोगः नाम दोष', 'आयुः वर्धनम्']
    noisy = ['योगः नाम द्रोष', 'आयुः वर्धनम्']
    r = full_report(clean, noisy, ['रोगः नाम दोष', 'आयु वर्धनम्'])          # fixes line 1, breaks line 2
    assert r['fully_fixed'] == 1.0 and r['over_correction'] == 1.0 and r['made_worse'] == 0.5
    assert r['types_before'] == {'consonant': 1, 'conjunct': 1} and r['types_after'] == {'mark': 1}
    lo, hi = bootstrap_cer(clean * 5, noisy * 5)
    assert 0 <= lo <= hi
    print('ok', r)
