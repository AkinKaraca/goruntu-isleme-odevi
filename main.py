import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from numba import set_num_threads, config
from cpu import CPU


def save(path, a):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, a)
    if not ok:
        raise OSError(f'Görüntü yazılamadı: {path}')
    encoded.tofile(str(path))


def read(path):
    a = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise ValueError(f'Görüntü okunamadı: {path}')
    return a


def examples(folder):
    rng = np.random.default_rng(2026)
    y, x = np.mgrid[:256, :384]
    clean = np.asarray(40 + x * 150 / 383, np.uint8)
    cv2.rectangle(clean, (30, 30), (140, 170), 210, -1)
    cv2.circle(clean, (245, 120), 65, 65, -1)
    cv2.putText(clean, 'OpenCV', (85, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 235, 2)
    low = np.asarray(95 + clean.astype(np.float64) * 45 / 255, np.uint8)
    noisy = np.clip(clean.astype(float) + rng.normal(0, 18, clean.shape), 0, 255).astype(np.uint8)
    salt = clean.copy()
    r = rng.random(clean.shape)
    salt[r < 0.05] = 0
    salt[r > 0.95] = 255
    for name, a in [('temiz', clean), ('dusuk_kontrast', low), ('gurultulu', noisy), ('tuz_biber', salt)]:
        save(folder / f'{name}.png', a)


def main():
    p = argparse.ArgumentParser(description='Beş görüntü işleme görevi')
    p.add_argument('--backend', choices=['serial', 'parallel', 'cuda'], default='parallel')
    p.add_argument('--input', type=Path)
    p.add_argument('--noisy', type=Path)
    p.add_argument('--salt', type=Path)
    p.add_argument('--out', type=Path, default=Path(__file__).parent / 'sonuclar')
    p.add_argument('--threads', type=int, default=min(8, os.cpu_count() or 1))
    args = p.parse_args()
    if not 1 <= args.threads <= config.NUMBA_NUM_THREADS:
        p.error(f'--threads 1..{config.NUMBA_NUM_THREADS} aralığında olmalı.')
    set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    if args.backend == 'cuda':
        from gpu import GPU
        engine = GPU()
    else:
        engine = CPU(args.backend == 'parallel')
    folder = args.out
    folder.mkdir(parents=True, exist_ok=True)
    examples(folder / 'girdiler')
    a = read(args.input or folder / 'girdiler/dusuk_kontrast.png')
    noisy = read(args.noisy or folder / 'girdiler/gurultulu.png')
    salt = read(args.salt or folder / 'girdiler/tuz_biber.png')
    hist, cdf, lut, lo, hi = engine.statistics(a)
    result = {'backend': args.backend, 'threads': args.threads, 'minimum': int(lo),
              'maximum': int(hi), 'opencv': cv2.__version__, 'timings_ms': {}}
    jobs = [('01_dogrusal', engine.linear, a), ('02_histogram_esitleme', engine.equalize, a),
            ('03_clahe_el', engine.clahe, a),
            ('03_clahe_opencv', cv2.createCLAHE(2.0, (8, 8)).apply, a),
            ('04_ortalama_3x3', engine.mean, noisy),
            ('05_medyan_el_5x5', engine.median, salt),
            ('05_medyan_opencv_5x5', lambda z: cv2.medianBlur(z, 5), salt)]
    images = {}
    for name, fn, src in jobs:
        fn(src)
        t = perf_counter()
        dst = fn(src)
        result['timings_ms'][name] = round(1000 * (perf_counter() - t), 3)
        save(folder / f'{name}.png', dst)
        images[name] = dst
    for kind, manual, reference in [('clahe', '03_clahe_el', '03_clahe_opencv'),
                                     ('median', '05_medyan_el_5x5', '05_medyan_opencv_5x5')]:
        diff = np.abs(images[manual].astype(np.int16) - images[reference].astype(np.int16))
        result[kind + '_max_difference'] = int(diff.max())
        result[kind + '_mean_difference'] = float(diff.mean())
    with (folder / 'histogram_cdf.csv').open('w', encoding='utf-8') as f:
        f.write('intensity,count,cumulative_count,cdf,equalized_value\n')
        for k in range(256):
            f.write(f'{k},{hist[k]},{cdf[k]},{cdf[k] / a.size:.10f},{lut[k]}\n')
    panels = [('Dusuk kontrast', a), ('Dogrusal', images['01_dogrusal']),
              ('Histogram esitleme', images['02_histogram_esitleme']), ('CLAHE', images['03_clahe_el']),
              ('Gurultulu', noisy), ('Ortalama 3x3', images['04_ortalama_3x3']),
              ('Tuz-biber', salt), ('Medyan 5x5', images['05_medyan_el_5x5'])]
    board = np.full((2 * 230, 4 * 300), 245, np.uint8)
    for i, (title, src) in enumerate(panels):
        y0, x0 = (i // 4) * 230, (i % 4) * 300
        cv2.putText(board, title, (x0 + 10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 20, 1)
        scale = min(280 / src.shape[1], 180 / src.shape[0])
        small = cv2.resize(src, (max(1, round(src.shape[1] * scale)), max(1, round(src.shape[0] * scale))))
        board[y0 + 40:y0 + 40 + small.shape[0], x0 + 10:x0 + 10 + small.shape[1]] = small
    save(folder / 'karsilastirma.png', board)
    (folder / 'rapor.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
