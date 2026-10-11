# Görüntü işleme ödevi

## Hocanın ek istediği C++ ve CUDA kurulumu

`opencv_cuda_colab.ipynb` dosyası Google Colab'da OpenCV C++ kitaplığını CUDA destekli derler, GPU işlemini doğrular ve bir dizindeki görüntüleri C++ ile **1024×768** boyutunda kaydeder. Notebook kaynakları ve örnekleri içerir. Kullanım: [CPP_COLAB_KULLANIM.md](CPP_COLAB_KULLANIM.md). C++ kaynakları `cpp/` dizinindedir. Yeni defterin gerçek Colab/GPU çalıştırması henüz doğrulanmamıştır; çıktı hücreleri bu nedenle boştur. Çalıştırılmış defter ve GPU doğrulama günlüğü sonradan teslim edilmelidir.

İlk beş görev aşağıdaki Python uygulamasındadır. Önceki Numba CUDA uygulaması ile yeni OpenCV CUDA kurulumu farklıdır. Hocanın paylaştığı metinde C++ şartı dizinden okuma ve boyutlandırma maddesinde yer almaktadır; ilk beş işlemin C++ olması ayrıca isteniyorsa bu dosyalar o şartı karşılamaz.

Kodlar istek doğrultusunda yorum satırı içermez. Python 3.12 önerilir. OpenCV görüntü okuma/yazma, istenen hazır CLAHE ve medyan işlemleri, örnek üretimi ve karşılaştırma için kullanılır. Elle yazılan algoritmalarda hazır histogram, CDF, normalizasyon, eşitleme veya filtre fonksiyonu çağrılmaz. NumPy dizileri bellek ve veri tipleri için, Numba açık döngüleri derlemek için kullanılır. Test dosyasındaki hazır fonksiyonlar yalnızca doğrulama içindir.

## Çalıştırma

Bu klasörde bir terminal açın:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --backend serial --out sonuclar_seri
python main.py --backend parallel --threads 4 --out sonuclar_paralel
python test_algorithms.py
```

Kendi görüntüleriniz:

```powershell
python main.py --input dusuk_kontrast.png --noisy gurultulu.png --salt tuz_biber.png --backend parallel
```

Giriş dosyaları OpenCV ile 8 bit gri olarak açılır. Dosya verilmediğinde sabit tohumla üretilen örnekler diske kaydedilip tekrar açılır. Tuz-biber örneği sentetiktir; yaklaşık %5 siyah ve %5 beyaz piksel içerir. İnternetten alınmış fotoğraf değildir. Harici görüntülerde düşük kontrast/gürültü niteliğini kullanıcı seçer.

## İşlemler

1. Histogramın ilk ve son dolu kutusundan minimum `m` ve maksimum `M` bulunur. Her piksel için `round(255*(I-m)/(M-m))` hesaplanır. Böylece minimum 0, maksimum 255 olur. `m=M` ise sıfıra bölmeyi önlemek için görüntü korunur.
2. 256 kutu piksel sayılarak doldurulur. Kümülatif sayı `C(k)=h(0)+...+h(k)`, normalize CDF `F(k)=C(k)/N` olur. İlk dolu seviyenin kümülatif sayısı `Cmin` alınır; eşleme `round(255*(C(k)-Cmin)/(N-Cmin))` biçimindedir. Sabit görüntü korunur. Bu, ilk dolu seviyeyi 0'a eşleyen CDF-min yaklaşımıdır. Hem histogram hem normalize CDF ve eşleme `histogram_cdf.csv` dosyasına yazılır.
3. Hazır sürüm `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` kullanır. Elle sürüm bölgesel histogram, kırpma, fazla sayıların yeniden dağıtımı, kümülatif eşleme ve dört komşu bölgenin çift doğrusal enterpolasyonunu uygular. Izgara değeri piksel boyutu değil bölge sayısıdır. Kenarlar REFLECT_101 mantığıyla genişletilir. Bölge kırpma sınırı `max(1, int(clipLimit*bolge_alani/256))` olur. Yeniden dağıtım sonrası bazı kutular bu sınırı aşabilir; yeniden kırpılmaz. CLAHE CDF'sinde global eşitlemedeki CDF-min çıkarımı yapılmaz. Kayan nokta yuvarlaması nedeniyle OpenCV ile 1 gri seviye fark kabul edilir.
4. 3×3 çekirdeğin dokuz katsayısı `1/9` olur. Konvolüsyon her pikselde açık döngülerle hesaplanır. Kenarda en yakın piksel tekrarlanır. Toplam geniş tamsayıda tutulur; uint8 taşması önlenir.
5. Hazır sürüm `cv2.medianBlur(I,5)` kullanır. Elle sürüm 5×5 komşuluğun 25 değerini ekleme sıralamasıyla sıralar ve 13. değeri alır. Kenarda en yakın piksel tekrarlanır. Bu davranış OpenCV medyan filtresiyle aynıdır.

## Çok iş parçacığı ve CUDA

`CPU(False)` seri, `CPU(True)` çok iş parçacıklı çalışır. Numba `prange` ile satırları veya CLAHE bölgelerini bağımsız iş parçacıklarına dağıtır. Histogramda her satırın ayrı 256 kutusu vardır; ortak histogram artırma yarışı oluşmaz. Son histogram birleştirmesi ve 256 elemanlı CDF küçük seri aşamalardır. Satır histogramları için yaklaşık `yükseklik*256*8` bayt ek bellek gerekir.

CUDA kurulumu ve çalıştırma:

```powershell
python -m pip install -r requirements-cuda.txt
python main.py --backend cuda --out sonuclar_cuda
python test_algorithms.py --cuda
```

Uyumlu NVIDIA GPU, sürücü ve numba-cuda kurulumuna uygun CUDA araç zinciri gerekir. Standart OpenCV paketinin CUDA ile derlenmesine gerek yoktur; GPU hesaplarını `gpu.py` içindeki özel Numba CUDA kernel'leri yapar. CUDA bulunamazsa program açık hata verir.

CUDA histogramı atomik artırma kullanır; CDF ve minimum/maksimum çıkarımı GPU üzerinde tek iş parçacığında yapılır. Dönüşümler ve filtreler piksel başına paraleldir. CLAHE histogram/LUT aşamasında bölge başına bir GPU iş parçacığı çalışır; enterpolasyon piksel başına paraleldir. Bu sürüm eğitim amaçlıdır: atomik histogram yoğunluğu ve bölge başına seri döngüler performansı sınırlayabilir. Her işlemin daha hızlı olduğu iddia edilmez. Histogram sayacı int32 olduğundan CUDA girdisi en fazla 2.147.483.647 pikseldir.

GPU olmadan algoritma mantığını simülatörde kontrol etmek için yeni bir terminal sürecinde:

```powershell
$env:NUMBA_ENABLE_CUDASIM="1"
python test_algorithms.py --cuda
Remove-Item Env:NUMBA_ENABLE_CUDASIM
```

Simülatör testi gerçek GPU derlemesini, sürücü uyumunu veya hızını doğrulamaz. Gerçek CUDA performansı için simülatör kapalı olmalıdır. Bazı numba-cuda sürümleri simülatör için de CUDA runtime kitaplığı ister. Bu Windows ortamında `nvidia-cuda-runtime-cu12` paketi kurularak bu eksiklik giderilmiştir. Denenen ortamın tam paket sürümleri `requirements-tested.txt` içindedir; gerçek GPU kurulumu araç zincirinize göre değişebilir.

`rapor.json` ilk ısınma çağrısından sonraki bir çalıştırmanın süresini içerir; dosya okuma/yazma hariçtir. CUDA ölçümü aygıt aktarımı, hesaplama ve sonucu geri alma süresini kapsar. İlk JIT derlemesi ölçüme alınmaz. Güvenilir performans karşılaştırması için aynı görüntülerle birkaç tekrar ve yeterince büyük girdiler kullanın.

## Dosyalar

- `cpu.py`: elle yazılmış seri ve çok iş parçacıklı algoritmalar.
- `gpu.py`: beş işlem için CUDA kernel'leri.
- `main.py`: örnekler, OpenCV karşılaştırmaları, kaydetme ve süre ölçümü.
- `test_algorithms.py`: histogram/CDF doğruluğu, backend eşliği, sabit/tek pikselli görüntüler, bölünmeyen boyutlar ve gürültü azaltma kontrolleri.
- `sonuclar/`: çalıştırılmış örnekler, yedi çıktı görüntüsü, görsel karşılaştırma, histogram/CDF ve rapor.

## Kaynaklar

- [OpenCV CLAHE arayüzü](https://docs.opencv.org/4.8.0/d6/dc7/group__imgproc__hist.html)
- [OpenCV CLAHE kaynak uygulaması: uyumluluk davranışları](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/clahe.cpp)
- [Numba paralel döngüler](https://numba.readthedocs.io/en/stable/user/parallel.html)
- [NVIDIA numba-cuda](https://github.com/NVIDIA/numba-cuda)
