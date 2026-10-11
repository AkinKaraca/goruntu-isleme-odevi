import argparse
import json
import numpy as np
import cv2
from numba import set_num_threads
from cpu import CPU


def test(gpu=False):
    set_num_threads(2)
    serial = CPU(False)
    other = CPU(True)
    if gpu:
        from gpu import GPU
        other = GPU()
    rng = np.random.default_rng(19)
    cases = [np.array([[123]], np.uint8), np.full((9, 17), 100, np.uint8),
             np.array([[0, 50, 100, 255]], np.uint8),
             rng.integers(90, 140, (31, 40), dtype=np.uint8),
             rng.integers(0, 256, (32, 35), dtype=np.uint8),
             rng.integers(0, 256, (32, 40), dtype=np.uint8),
             rng.integers(0, 256, (1, 19), dtype=np.uint8),
             rng.integers(0, 256, (19, 1), dtype=np.uint8)]
    checks = 0
    worst_clahe = 0
    for a in cases:
        hist, cdf, _, lo, hi = serial.statistics(a)
        assert np.array_equal(hist, np.bincount(a.ravel(), minlength=256))
        assert np.array_equal(cdf, np.cumsum(hist))
        assert lo == int(a.min()) and hi == int(a.max())
        checks += 3
        b = serial.linear(a)
        if lo < hi:
            assert b.min() == 0 and b.max() == 255
        else:
            assert np.array_equal(a, b)
        checks += 1
        assert np.max(np.abs(serial.equalize(a).astype(int) - cv2.equalizeHist(a).astype(int))) <= 1
        assert np.array_equal(serial.median(a), cv2.medianBlur(a, 5))
        assert np.array_equal(serial.mean(a), cv2.blur(a, (3, 3), borderType=cv2.BORDER_REPLICATE))
        checks += 3
        for method in ('linear', 'equalize', 'mean', 'median'):
            assert np.array_equal(getattr(serial, method)(a), getattr(other, method)(a)), method
            checks += 1
        stats = other.statistics(a)
        assert np.array_equal(stats[0], hist) and np.array_equal(stats[1], cdf)
        checks += 1
        for grid in ((8, 8), (3, 2)):
            for clip in (0.0, 2.0, 10.0):
                manual = serial.clahe(a, clip, grid)
                ref = cv2.createCLAHE(clip, grid).apply(a)
                diff = int(np.abs(manual.astype(int) - ref.astype(int)).max())
                worst_clahe = max(worst_clahe, diff)
                assert diff <= 1, ('CLAHE', a.shape, clip, grid, diff)
                diff2 = int(np.abs(manual.astype(int) - other.clahe(a, clip, grid).astype(int)).max())
                assert diff2 <= (1 if gpu else 0), ('backend CLAHE', diff2)
                checks += 2
    assert np.array_equal(serial.linear(np.array([[100, 110, 120]], np.uint8)), [[0, 128, 255]])
    assert np.array_equal(serial.equalize(np.array([[10, 10, 20, 30]], np.uint8)), [[0, 0, 128, 255]])
    clean = np.full((64, 64), 128, np.uint8)
    noise = np.clip(clean.astype(float) + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)
    salt = clean.copy()
    r = rng.random(clean.shape)
    salt[r < .05], salt[r > .95] = 0, 255
    mse = lambda a: float(np.mean((a.astype(float) - clean) ** 2))
    assert mse(serial.mean(noise)) < mse(noise)
    assert mse(serial.median(salt)) < mse(salt)
    checks += 4
    print(json.dumps({'checks_passed': checks, 'backend': 'cuda' if gpu else 'parallel',
                      'clahe_max_difference_opencv': worst_clahe}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cuda', action='store_true')
    test(p.parse_args().cuda)
