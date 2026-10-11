import math
import numpy as np
from numba import int32
try:
    from numba import cuda
except Exception as exc:
    raise RuntimeError('CUDA yüklenemedi. NVIDIA sürücüsünü ve CUDA araç zincirini kontrol edin; CPU için --backend parallel kullanın.') from exc
from cpu import validate, geometry


@cuda.jit
def histogram_kernel(a, hist):
    x, y = cuda.grid(2)
    if y < a.shape[0] and x < a.shape[1]:
        cuda.atomic.add(hist, int(a[y, x]), 1)


@cuda.jit
def distribution_kernel(hist, cdf, lut, bounds):
    if cuda.grid(1) == 0:
        acc, first, lo, hi = 0, 0, 255, 0
        for k in range(256):
            acc += hist[k]
            cdf[k] = acc
            if hist[k] > 0:
                if first == 0:
                    first, lo = acc, k
                hi = k
        bounds[0], bounds[1] = lo, hi
        for k in range(256):
            lut[k] = 0
            if acc == first:
                lut[k] = k
            elif cdf[k] >= first:
                lut[k] = round(255.0 * (cdf[k] - first) / (acc - first))


@cuda.jit
def mapping_kernel(a, out, lut, bounds, linear):
    x, y = cuda.grid(2)
    if y < a.shape[0] and x < a.shape[1]:
        v = int(a[y, x])
        if linear:
            lo, hi = bounds[0], bounds[1]
            out[y, x] = v if hi == lo else round((v - lo) * 255.0 / (hi - lo))
        else:
            out[y, x] = lut[v]


@cuda.jit
def filter_kernel(a, out, median):
    x, y = cuda.grid(2)
    h, w = a.shape
    if y < h and x < w:
        if median:
            values = cuda.local.array(25, int32)
            n = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    yy = min(max(y + dy, 0), h - 1)
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
            acc = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    yy = min(max(y - dy, 0), h - 1)
                    xx = min(max(x - dx, 0), w - 1)
                    acc += int(a[yy, xx])
            out[y, x] = (acc + 4) // 9


@cuda.jit
def tile_kernel(a, luts, th, tw, clip):
    t = cuda.grid(1)
    gy, gx, _ = luts.shape
    h, w = a.shape
    if t < gx * gy:
        ty, tx = t // gx, t % gx
        hist = cuda.local.array(256, int32)
        for k in range(256):
            hist[k] = 0
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


@cuda.jit
def interpolation_kernel(a, out, luts, th, tw):
    x, y = cuda.grid(2)
    gy, gx, _ = luts.shape
    if y < a.shape[0] and x < a.shape[1]:
        fy = np.float32(y) * np.float32(1.0 / th) - np.float32(0.5)
        fx = np.float32(x) * np.float32(1.0 / tw) - np.float32(0.5)
        iy, ix = math.floor(fy), math.floor(fx)
        ay, ax = np.float32(fy - iy), np.float32(fx - ix)
        y0, y1 = max(iy, 0), min(iy + 1, gy - 1)
        x0, x1 = max(ix, 0), min(ix + 1, gx - 1)
        bx, by = np.float32(1) - ax, np.float32(1) - ay
        v = int(a[y, x])
        top = luts[y0, x0, v] * bx + luts[y0, x1, v] * ax
        bot = luts[y1, x0, v] * bx + luts[y1, x1, v] * ax
        out[y, x] = min(255, max(0, round(top * by + bot * ay)))


class GPU:
    def __init__(self):
        if not cuda.is_available():
            raise RuntimeError('CUDA aygıtı bulunamadı. CPU için --backend parallel kullanın.')

    def prepare(self, a):
        a = validate(a)
        if a.size > 2147483647:
            raise ValueError('CUDA histogramı için görüntü çok büyük.')
        blocks = ((a.shape[1] + 15) // 16, (a.shape[0] + 15) // 16)
        return cuda.to_device(a), cuda.device_array(a.shape, np.uint8), blocks

    def stats_device(self, a, blocks):
        hist = cuda.to_device(np.zeros(256, np.int32))
        cdf = cuda.device_array(256, np.int32)
        lut = cuda.device_array(256, np.uint8)
        bounds = cuda.device_array(2, np.int32)
        histogram_kernel[blocks, (16, 16)](a, hist)
        distribution_kernel[1, 1](hist, cdf, lut, bounds)
        return hist, cdf, lut, bounds

    def statistics(self, a):
        d, _, blocks = self.prepare(a)
        hist, cdf, lut, bounds = self.stats_device(d, blocks)
        b = bounds.copy_to_host()
        return hist.copy_to_host(), cdf.copy_to_host(), lut.copy_to_host(), int(b[0]), int(b[1])

    def mapped(self, a, linear):
        d, out, blocks = self.prepare(a)
        _, _, lut, bounds = self.stats_device(d, blocks)
        mapping_kernel[blocks, (16, 16)](d, out, lut, bounds, linear)
        return out.copy_to_host()

    def linear(self, a):
        return self.mapped(a, True)

    def equalize(self, a):
        return self.mapped(a, False)

    def clahe(self, a, clip=2.0, grid=(8, 8)):
        if not math.isfinite(clip) or clip < 0:
            raise ValueError('clip sonlu ve negatif olmayan bir sayı olmalı.')
        d, out, blocks = self.prepare(a)
        th, tw = geometry(a, grid)
        gx, gy = grid
        luts = cuda.device_array((gy, gx, 256), np.uint8)
        tile_kernel[(gx * gy + 63) // 64, 64](d, luts, th, tw, clip)
        interpolation_kernel[blocks, (16, 16)](d, out, luts, th, tw)
        return out.copy_to_host()

    def filtered(self, a, median):
        d, out, blocks = self.prepare(a)
        filter_kernel[blocks, (16, 16)](d, out, median)
        return out.copy_to_host()

    def mean(self, a):
        return self.filtered(a, False)

    def median(self, a):
        return self.filtered(a, True)
