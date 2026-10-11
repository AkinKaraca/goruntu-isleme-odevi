import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request


def run(args, log):
    print(' '.join(map(str, args)), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(encoding='utf-8', errors='replace')
    print(text[-5000:], flush=True)
    if result.returncode:
        raise RuntimeError(f'Komut basarisiz. Tam gunluk: {log}')
    return text


def install(root=Path('/content/opencv_odevi')):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not shutil.which('nvidia-smi') or not shutil.which('nvcc'):
        raise RuntimeError('Colab GPU oturumunu secin. nvidia-smi ve nvcc bulunmali.')
    run(['nvidia-smi'], root / 'nvidia-smi.txt')
    run(['nvcc', '--version'], root / 'nvcc.txt')
    archs = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True).splitlines()
    arch = archs[0].strip() if archs else ''
    if not re.fullmatch(r'\d+\.\d+', arch):
        raise RuntimeError('GPU compute capability okunamadi; CUDA_ARCH_BIN otomatik secilemedi.')
    run(['apt-get', 'update', '-qq'], root / 'apt-update.log')
    run(['apt-get', 'install', '-y', '-qq', 'build-essential', 'cmake', 'ninja-build',
         'libjpeg-dev', 'libpng-dev', 'libtiff-dev', 'libwebp-dev'], root / 'apt-install.log')
    version = '4.14.0'
    for name in ('opencv', 'opencv_contrib'):
        source = root / f'{name}-{version}'
        if not source.is_dir():
            archive = root / f'{name}-{version}.tar.gz'
            urllib.request.urlretrieve(f'https://github.com/opencv/{name}/archive/refs/tags/{version}.tar.gz', archive)
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter='data')
    prefix = root / 'install'
    build = root / 'opencv-build'
    config = run([
        'cmake', '-S', root / f'opencv-{version}', '-B', build, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_INSTALL_PREFIX={prefix}',
        f'-DOPENCV_EXTRA_MODULES_PATH={root / f"opencv_contrib-{version}" / "modules"}',
        '-DWITH_CUDA=ON', f'-DCUDA_ARCH_BIN={arch}', '-DCUDA_ARCH_PTX=',
        '-DBUILD_LIST=core,imgproc,imgcodecs,cudawarping', '-DBUILD_SHARED_LIBS=ON',
        '-DBUILD_TESTS=OFF', '-DBUILD_PERF_TESTS=OFF', '-DBUILD_EXAMPLES=OFF',
        '-DBUILD_opencv_python2=OFF', '-DBUILD_opencv_python3=OFF', '-DBUILD_JAVA=OFF',
        '-DWITH_CUDNN=OFF', '-DOPENCV_DNN_CUDA=OFF', '-DWITH_CUBLAS=OFF',
        '-DWITH_CUFFT=OFF', '-DWITH_NVCUVID=OFF', '-DWITH_NVCUVENC=OFF',
        '-DWITH_IPP=OFF', '-DWITH_OPENCL=OFF', '-DWITH_FFMPEG=OFF',
        '-DWITH_GSTREAMER=OFF', '-DWITH_GTK=OFF', '-DWITH_QT=OFF',
        '-DOPENCV_GENERATE_PKGCONFIG=ON'
    ], root / 'opencv-configure.log')
    if not re.search(r'NVIDIA CUDA:\s+YES', config):
        raise RuntimeError('CMake NVIDIA CUDA: YES raporlamadi. CPU derlemesine devam edilmiyor.')
    jobs = str(min(2, os.cpu_count() or 1))
    run(['cmake', '--build', build, '--parallel', jobs], root / 'opencv-build.log')
    run(['cmake', '--install', build], root / 'opencv-install.log')
    print(f'OpenCV C++ CUDA kurulumu tamamlandi: {prefix}', flush=True)
    return prefix


if __name__ == '__main__':
    install()
