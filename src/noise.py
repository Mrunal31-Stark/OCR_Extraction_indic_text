"""Synthetic OCR noise for Devanagari, calibrated on real Tesseract errors (see notebook Phase 1).

Error mix in real OCR on my test pages: ~60% look-alike consonant swaps, ~25% mark (matra/anusvara/visarga)
errors, ~10% conjunct insert/drop, ~5% spacing/punctuation. `corrupt()` reproduces that mix.
"""
import random
import unicodedata

# Look-alike glyph confusions. First entries were observed in Phase 1; the rest are well-known Devanagari OCR confusions.
CONSONANT_CONFUSIONS = {
    'य': ['म', 'थ', 'ग', 'त्र'], 'व': ['ठ', 'त', 'क', 'न', 'ब'], 'ह': ['द'], 'स': ['य', 'म'], 'ष': ['ण', 'प'],
    'र': ['य'], 'म': ['य', 'भ'], 'द': ['ह', 'ढ', 'द्र'], 'प': ['ष', 'य'], 'ब': ['व'], 'घ': ['ध'], 'ध': ['घ'],
    'भ': ['म'], 'ट': ['ठ'], 'ठ': ['ट', 'व'], 'न': ['त'], 'त': ['न'], 'ण': ['ष'], 'ख': ['रव'], 'थ': ['य'],
    'ञ': ['ज'], 'झ': ['भ'], 'ङ': ['ड'], 'ड': ['ङ'], 'छ': ['ढ'], 'क': ['फ'], 'फ': ['क'],
}
MARK_CONFUSIONS = {
    'ं': ['े', ''], 'ः': ['', ':'], 'ा': [''], 'ि': ['ी', ''], 'ी': ['ि'], 'ु': ['ू', ''], 'ू': ['ु'],
    'े': ['ै', ''], 'ै': ['े'], 'ो': ['ौ', 'ा'], 'ौ': ['ो', ''], 'ृ': ['ु'], '्': [''], 'ँ': ['ं'],
}
PUNCT_CONFUSIONS = {'।': ['|', '', 'l'], '॥': ['||', '।', '|'], ' ': ['']}  # ' ' -> '' merges words
CONSONANTS = set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')

# op -> relative weight, from the Phase 1 error mix
OP_WEIGHTS = {'consonant': 0.60, 'mark': 0.25, 'conjunct': 0.10, 'punct_space': 0.05}


def corrupt(text, rate, rng=random):
    """Return an OCR-like noisy version of `text`. `rate` ~ expected fraction of characters edited."""
    out, chars, i = [], list(text), 0
    while i < len(chars):
        c = chars[i]
        if rng.random() >= rate:
            out.append(c); i += 1; continue
        op = rng.choices(list(OP_WEIGHTS), weights=list(OP_WEIGHTS.values()))[0]
        if op == 'consonant' and c in CONSONANT_CONFUSIONS:
            out.append(rng.choice(CONSONANT_CONFUSIONS[c]))
        elif op == 'mark' and c in MARK_CONFUSIONS:
            out.append(rng.choice(MARK_CONFUSIONS[c]))
        elif op == 'conjunct' and c in CONSONANTS:
            nxt = chars[i + 1:i + 3]
            if nxt[:1] == ['्'] and len(nxt) == 2:            # C + virama + C  -> drop the half-letter
                out.append(c); i += 3; continue
            out.append(c + '्र')                               # spurious ra-phala (seen: दोष -> द्रोष)
        elif op == 'punct_space' and c in PUNCT_CONFUSIONS:
            out.append(rng.choice(PUNCT_CONFUSIONS[c]))
        elif op == 'punct_space' and c in CONSONANTS and rng.random() < 0.3:
            out.append(c + ' ')                                # spurious word split
        else:
            out.append(c)                                      # op not applicable to this char
        i += 1
    return unicodedata.normalize('NFC', ' '.join(''.join(out).split()))


def sample_rate(rng=random):
    """Per-line noise level: 10% clean lines (teach 'leave correct text alone'), rest spread around real OCR's ~10% CER.
    Measured: effective CER ~= rate/2, so rate in [0.03, 0.45] -> CER ~1.5%-20% (real Tesseract on my pages: 9.6%)."""
    return 0.0 if rng.random() < 0.10 else rng.uniform(0.03, 0.45)


if __name__ == '__main__':
    rng = random.Random(0)
    s = 'रोगः नाम दोष वैषम्य कारणम् दोष सम्यता आरोग्य कारणम्'
    assert corrupt(s, 0.0, rng) == s
    noisy = [corrupt(s, 0.15, rng) for _ in range(200)]
    assert sum(n != s for n in noisy) > 150 and all(unicodedata.is_normalized('NFC', n) for n in noisy)
    print('ok', noisy[:3])
