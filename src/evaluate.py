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


if __name__ == '__main__':
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
