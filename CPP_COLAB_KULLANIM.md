# Hocanın ek görevi: CUDA OpenCV kurulumu ve C++ boyutlandırma

## En kolay yol: Google Colab

1. `opencv_cuda_colab.ipynb` dosyasını [Google Colab](https://colab.research.google.com/) içinde **Dosya → Not defteri yükle** ile açın.
2. **Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU** seçin.
3. Hücreleri sırayla çalıştırın. OpenCV kaynak koddan derlendiği için kurulum uzun sürebilir. Oturum kapanırsa geçici dosyalar kaybolabilir.
4. Kurulum doğrulamasında `NVIDIA CUDA: YES`, sıfırdan büyük `CUDA_DEVICE_COUNT` ve `CUDA_RESIZE_TEST=PASS` beklenir. Program yalnızca `cv2.cuda` isminin varlığına bakmaz; C++ ile gerçek GPU işlemi yapar.
5. C++ programı örnekleri seri CPU, çok iş parçacıklı CPU ve GPU üzerinde işler. Her görüntü **1024 piksel genişlik ve 768 piksel yükseklik** ile PNG olarak kaydedilir.
6. Son hücrede sonuç ZIP'ini indirin. Defterin çalıştırılmış halini de **Dosya → İndir → .ipynb indir** üzerinden kaydedin.
7. Çalıştırılmış `.ipynb` dosyasını GitHub'daki aynı dosyanın üzerine yükleyin. Sonuçlara `cuda_dogrulama.txt` ve C++ test raporunu da ekleyin. Böylece kurulumun çalıştığına dair gerçek kanıt bulunur.

Defter kendi kaynaklarını ve örnek görüntülerini içerir; depodaki önceki görüntülerin hangi klasöre yüklendiğine bağımlı değildir. Harici resim kullanmak için ilgili hücrede `KENDI_GORUNTULERIMI_YUKLE = True` yapın. Kaynağı dışarıdan alınan bir resim kullanırsanız kaynağını da teslimde belirtin.

### Defter hangi OpenCV'yi kuruyor?

C++ görevini karşılayan **OpenCV 4.14.0 C++ SDK** ve aynı sürüm `opencv_contrib` derlenir. `WITH_CUDA=ON` kullanılır, gerçek GPU mimarisi `nvidia-smi` ile belirlenir. `core`, `imgproc`, `imgcodecs`, `cudawarping` ve gerekli bağımlı modüller derlenir. Python bağlayıcıları bu derlemede kapalıdır. Colab'da önceden kurulu Python `cv2`, çıktıların okunması/doğrulanması için kullanılır; CUDA'lı OpenCV kurulumu olarak sunulmaz. C++ `cuda_kontrol` programı yeni kurulan kitaplığa bağlanır.

GPU/araç zinciri yoksa defter açık hata vererek durur. NVIDIA sürücüsü, CUDA araç zinciri veya Colab kaynak sınırlarından doğan hatalar CPU'ya sessizce geçilerek saklanmaz. Ücretsiz Colab GPU tahsisi ve oturum süresi garanti değildir.

## C++ programının davranışı

- Girdi dizininin doğrudan içindeki PNG, JPG/JPEG, BMP, TIFF, WebP, PGM ve PPM dosyalarını bulur. Alt dizinler taranmaz.
- `cv::imdecode` ile dosyayı okur; UTF-8 dosya yolları için C++ dosya akışı kullanır.
- CPU'da `cv::resize`, GPU'da `cv::cuda::resize` çağırır. Enterpolasyon `INTER_LINEAR` seçilmiştir.
- Çıktı 1024×768'e tam olarak esnetilir; en-boy oranının korunması istenmediği için kenarlık eklenmez.
- 8/16 bit, 1/3/4 kanallı görüntüleri PNG olarak korur. Başka veri tipleri hata olarak bildirilir.
- `fotograf.jpg` dosyası `fotograf.jpg.png` olur; aynı adlı JPG ve PNG girdileri birbirini ezmez.
- Bozuk/okunamayan görüntüler bildirilerek atlanır. Var olan çıktıların üzerine yazılmaz. Tekrar denemede başka çıktı klasörü seçilir.
- `--threads 4` görüntüleri dört C++ iş parçacığına dağıtır. Her bir resim ayrı işlenir. GPU modu tek iş parçacığıyla varsayılan olarak çalıştırılır.
- Dönüş kodları: `0` en az bir başarılı çıktı ve yazma/işleme hatası yok; `1` argüman/kurulum/dizin sorunu; `2` görüntü adayı var ama hiçbiri okunamadı; `3` en az bir dosyada işleme/yazma hatası. Atlanan bozuk dosyalar ayrıca raporlanır.

## Yerel C++ çalıştırma

VS Code bir editördür; ayrıca C++17 derleyicisi, CMake ve OpenCV C++ SDK gerekir. CUDA için bunlara uyumlu NVIDIA GPU/sürücüsü/Toolkit ve CUDA ile derlenmiş OpenCV eklenir. `pip install opencv-python`, C++ SDK ve CUDA OpenCV kurulumu yerine geçmez.

CPU derlemesi:

```text
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=OPENCV_CMAKE_DIZINI
cmake --build build --config Release
```

Linux:

```text
./build/boyutlandir girdiler ciktilar --threads 4
```

Windows Visual Studio derleyicisinde **Developer PowerShell for VS** açın. Çok yapılandırmalı Visual Studio derlemesinde:

```powershell
.\build\Release\boyutlandir.exe girdiler ciktilar --threads 4
```

Ninja derlemesinde yürütülebilir dosya doğrudan `build` altında olur. Windows'ta OpenCV DLL dizini PATH içinde bulunmalıdır.

CUDA ile derleme ve çalıştırma:

```text
cmake -S cpp -B build-cuda -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DOpenCV_DIR=CUDA_OPENCV_CMAKE_DIZINI
cmake --build build-cuda --config Release
./build-cuda/cuda_kontrol
./build-cuda/boyutlandir girdiler ciktilar_cuda --cuda
```

CPU otomatik testleri için Python bağımlılıkları kurulduktan sonra:

```text
python cpp/test_resize.py build/boyutlandir --out cpp_test_sonuclari
```

CUDA'lı derlemede aynı teste `--cuda` ekleyin. Windows'ta yürütülebilir yolunu kendi `.exe` konumunuza göre değiştirin.

## Kaynaklar

- [OpenCV yapılandırma seçenekleri](https://docs.opencv.org/4.12.0/db/d05/tutorial_config_reference.html)
- [OpenCV contrib](https://github.com/opencv/opencv_contrib)
- [OpenCV 4.14.0 sürümü](https://github.com/opencv/opencv/releases/tag/4.14.0)
- [Colab sık sorulan sorular](https://research.google.com/colaboratory/faq.html)
