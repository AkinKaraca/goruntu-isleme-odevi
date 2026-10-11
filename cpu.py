import math
import numpy as np
from numba import njit, prange


def validate(a):
    if a.ndim != 2 or a.dtype != np.uint8 or a.size == 0:
        raise ValueError('Boş olmayan, tek kanallı uint8 görüntü gerekli.')
    return np.ascontiguousarray(a)


def geometry(a, grid):
    gx, gy = grid
    if gx < 1 or gy < 1:
        raise ValueError('Izgara boyutları pozitif olmalı.')
    h, w = a.shape
    if h % gy or w % gx:
        h += gy - h % gy
        w += gx - w % gx
    return h // gy, w // gx


def _hist_rows(a):
    h, w = a.shape
    rows = np.zeros((h, 256), np.int64)
    for y in prange(h):
        for x in range(w):
            rows[y, int(a[y, x])] += 1
    return rows


@njit(cache=True)
def _distribution(rows):
    hist = np.zeros(256, np.int64)
    cdf = np.zeros(256, np.int64)
    for y in range(rows.shape[0]):
        for k in range(256):
            hist[k] += rows[y, k]
    total = 0
    lo, hi = 255, 0
    first = 0
    for k in range(256):
        total += hist[k]
        cdf[k] = total
        if hist[k] > 0:
            if first == 0:
                first = total
                lo = k
            hi = k
    lut = np.zeros(256, np.uint8)
    for k in range(256):
        if total == first:
            lut[k] = k
        elif cdf[k] >= first:
            lut[k] = round(255.0 * (cdf[k] - first) / (total - first))
    return hist, cdf, lut, lo, hi


def _mapping(a, lut):
    out = np.empty_like(a)
    for y in prange(a.shape[0]):
        for x in range(a.shape[1]):
            out[y, x] = lut[int(a[y, x])]
    return out


def _stretch(a, lo, hi):
    out = np.empty_like(a)
    for y in prange(a.shape[0]):
        for x in range(a.shape[1]):
            out[y, x] = a[y, x] if hi == lo else round(
                (int(a[y, x]) - lo) * 255.0 / (hi - lo))
    return out


def _filter(a, median):
    h, w = a.shape
    out = np.empty_like(a)
    for y in prange(h):
        values = np.empty(25, np.int32)
        for x in range(w):
            if median:
                n = 0
                for dy in range(-2, 3):
                    yy = min(max(y + dy, 0), h - 1)
                    for dx in range(-2, 3):
                        xx = min(max(x + dx, 0), w - 1)
                        v = int(a[yy, xx])
                        j = n - 1
                        while j >= 0 and values[j] > v:
                            values[j + 1] = values[j]
                            j -= 1
                        values[j + 1] = v
                        n += 1
                out[y, x] = values[12]
            else:
                s = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy = min(max(y - dy, 0), h - 1)
                        xx = min(max(x - dx, 0), w - 1)
                        s += int(a[yy, xx])
                out[y, x] = (s + 4) // 9
    return out


def _tile_luts(a, gx, gy, th, tw, clip):
    h, w = a.shape
    luts = np.empty((gy, gx, 256), np.uint8)
    for t in prange(gx * gy):
        ty, tx = t // gx, t % gx
        hist = np.zeros(256, np.int64)
        for dy in range(th):
            y = ty * th + dy
            yy = 0 if h == 1 else y % (2 * h - 2)
            if yy >= h:
                yy = 2 * h - 2 - yy
            for dx in range(tw):
                x = tx * tw + dx
                xx = 0 if w == 1 else x % (2 * w - 2)
                if xx >= w:
                    xx = 2 * w - 2 - xx
                hist[int(a[yy, xx])] += 1
        if clip > 0:
            limit = max(1, int(clip * th * tw / 256))
            excess = 0
            for k in range(256):
                if hist[k] > limit:
                    excess += hist[k] - limit
                    hist[k] = limit
            batch, rest = excess // 256, excess % 256
            for k in range(256):
                hist[k] += batch
            if rest:
                step = max(1, 256 // rest)
                for k in range(rest):
                    hist[k * step] += 1
        acc = 0
        scale = np.float32(255.0 / (th * tw))
        for k in range(256):
            acc += hist[k]
            luts[ty, tx, k] = min(255, round(np.float32(acc) * scale))
    return luts


def _interpolate(a, luts, th, tw):
    gy, gx, _ = luts.shape
    out = np.empty_like(a)
    for y in prange(a.shape[0]):
        fy = np.float32(y) * np.float32(1.0 / th) - np.float32(0.5)
        iy = math.floor(fy)
        ay = np.float32(fy - iy)
        y0, y1 = max(iy, 0), min(iy + 1, gy - 1)
        for x in range(a.shape[1]):
            fx = np.float32(x) * np.float32(1.0 / tw) - np.float32(0.5)
            ix = math.floor(fx)
            ax = np.float32(fx - ix)
            x0, x1 = max(ix, 0), min(ix + 1, gx - 1)
            v = int(a[y, x])
            bx, by = np.float32(1) - ax, np.float32(1) - ay
            top = luts[y0, x0, v] * bx + luts[y0, x1, v] * ax
            bot = luts[y1, x0, v] * bx + luts[y1, x1, v] * ax
            out[y, x] = min(255, max(0, round(top * by + bot * ay)))
    return out


class CPU:
    def __init__(self, parallel=False):
        for name in ('hist_rows', 'mapping', 'stretch', 'filter',
                     'tile_luts', 'interpolate'):
            setattr(self, name, njit(parallel=parallel)(globals()['_' + name]))

    def statistics(self, a):
        return _distribution(self.hist_rows(validate(a)))

    def linear(self, a):
        _, _, _, lo, hi = self.statistics(a)
        return self.stretch(a, lo, hi)

    def equalize(self, a):
        _, _, lut, _, _ = self.statistics(a)
        return self.mapping(a, lut)

    def clahe(self, a, clip=2.0, grid=(8, 8)):
        validate(a)
        if not math.isfinite(clip) or clip < 0:
            raise ValueError('clip sonlu ve negatif olmayan bir sayı olmalı.')
        th, tw = geometry(a, grid)
        lut = self.tile_luts(a, grid[0], grid[1], th, tw, clip)
        return self.interpolate(a, lut, th, tw)

    def mean(self, a):
        return self.filter(validate(a), False)

    def median(self, a):
        return self.filter(validate(a), True)
