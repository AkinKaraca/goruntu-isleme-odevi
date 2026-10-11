import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import cv2
import numpy as np


def save(path, a):
    ok, data = cv2.imencode(path.suffix, a)
    assert ok
    data.tofile(str(path))


def read(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)


def test(exe, destination, cuda=False):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='cpp_test_', dir=destination))
    inp = root / 'girdiler'
    inp.mkdir()
    rng = np.random.default_rng(42)
    samples = {
        'renk.png': rng.integers(0, 256, (123, 217, 3), dtype=np.uint8),
        'gri.png': rng.integers(0, 256, (59, 83), dtype=np.uint8),
        'derin.png': rng.integers(0, 65536, (91, 101), dtype=np.uint16),
        'alpha.png': rng.integers(0, 256, (27, 53, 4), dtype=np.uint8),
        'ayni.png': np.full((20, 30), 81, np.uint8),
        'ayni.jpg': np.full((30, 20), 201, np.uint8),
        'buyuk.PNG': np.full((900, 1400, 3), 123, np.uint8)
    }
    for name, image in samples.items():
        save(inp / name, image)
    (inp / 'bozuk.png').write_bytes(b'not an image')
    (inp / 'metin.txt').write_text('image olmayan dosya', encoding='utf-8')
    (inp / 'alt_dizin').mkdir()
    save(inp / 'alt_dizin' / 'dahil_degil.png', np.zeros((2, 2), np.uint8))
    commands = []

    def run(args, expected=0):
        p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, encoding='utf-8', errors='replace')
        commands.append({'args': list(map(str, args)), 'exit_code': p.returncode,
                         'stdout': p.stdout, 'stderr': p.stderr})
        assert p.returncode == expected, p.stdout + p.stderr

    for mode, args in [('serial', []), ('parallel', ['--threads', '4'])] + ([('cuda', ['--cuda'])] if cuda else []):
        out = root / mode
        run([inp, out, *args])
        assert len(list(out.glob('*.png'))) == len(samples)
        for name in samples:
            source = read(inp / name)
            result = read(out / (name + '.png'))
            assert result is not None and result.shape[:2] == (768, 1024)
            assert result.dtype == source.dtype and result.ndim == source.ndim
            if mode != 'cuda':
                ref = cv2.resize(source, (1024, 768), interpolation=cv2.INTER_LINEAR)
                assert np.abs(result.astype(float) - ref.astype(float)).max() <= 1
            if mode == 'parallel':
                assert np.array_equal(result, read(root / 'serial' / (name + '.png')))
        if mode == 'cuda':
            assert np.all(read(out / 'buyuk.PNG.png') == 123)
    run([inp, root / 'serial'], 3)
    run([inp, inp], 1)
    run([root / 'olmayan', root / 'yeni'], 1)
    run([inp, root / 'bad_threads', '--threads', '0'], 1)
    empty = root / 'bos'
    empty.mkdir()
    run([empty, root / 'bos_cikti'], 1)
    if not cuda:
        run([inp, root / 'cuda_yok', '--cuda'], 1)
    report = {'status': 'PASS', 'cuda_tested': cuda, 'images_per_mode': len(samples),
              'dimensions': [1024, 768], 'commands': commands}
    (destination / 'cpp_test_raporu.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'commands'}, indent=2))
    return root


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('executable', type=Path)
    p.add_argument('--out', type=Path, default=Path('cpp_test_sonuclari'))
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()
    test(args.executable.resolve(), args.out, args.cuda)
