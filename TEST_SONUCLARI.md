# Doğrulama sonuçları

11 Ekim 2026, Windows, Python 3.12.15, OpenCV 4.14.0, Numba 0.68.0.

- Seri/çok iş parçacıklı karşılaştırma: 196 kontrol başarılı.
- CUDA simülatörü/seri karşılaştırma: 196 kontrol başarılı.
- Test matrislerinde elle CLAHE ile OpenCV arasında en büyük fark: 0.
- Örnek çalıştırmada CLAHE ve 5×5 medyan için OpenCV ile en büyük fark: 0.
- Örnek düşük kontrastlı görüntü: minimum 102, maksimum 136; doğrusal çıktı aralığı 0–255.
- Dört Python kaynak dosyası tokenize edilerek kontrol edildi: 0 yorum satırı.
- Görsel karşılaştırma dosyası açılarak incelendi.

Kullanılabilir CUDA aygıtı bulunmadı (`cuda.is_available() == False`). CUDA testleri `NUMBA_ENABLE_CUDASIM=1` ile yapıldı. Gerçek GPU derlemesi, donanımdaki sonuçlar ve GPU hızlanması doğrulanmadı. Simülatör testi bunun yerine geçmez.

CPU örnek çalıştırmasının süreleri `sonuclar/rapor.json` dosyasındadır. Bu tek ölçüm, genel bir hızlanma iddiası değildir. Aynı dosyada OpenCV karşılaştırma farkları bulunur.

## Ek C++ görevi

- `cpp/resize_images.cpp` MSVC 19.51, C++17 ve resmi OpenCV 4.14.0 Windows SDK ile başarıyla derlendi.
- Yedi girdi türünde seri ve dört iş parçacıklı CPU çalıştırmaları test edildi: renkli, gri, 16 bit, alfa kanallı, aynı köklü JPG/PNG ve büyük harfli uzantı.
- Her çıktı 1024×768; seri ve çok iş parçacıklı sonuçlar birebir aynı. Python OpenCV referansına karşı en fazla 1 gri seviye toleranslı karşılaştırmalar geçti.
- Bozuk dosyanın atlanması; metin/alt klasörün tarama dışında tutulması; boş veya olmayan dizin, geçersiz iş parçacığı sayısı, aynı girdi/çıktı dizini ve var olan dosya üzerine yazmanın reddi test edildi.
- CPU derlemesinde `--cuda` seçeneğinin açık hata vermesi doğrulandı.
- Dört ödev örneği C++ programıyla boyutlandırıldı. Görseller `cpp_sonuclar/` içinde; otomatik test raporu aynı dizinde `cpp_test_raporu.json` dosyasındadır.
- `opencv_cuda_colab.ipynb` hücrelerinin Python sözdizimi ve gömülü kaynakların teslim kaynaklarıyla aynı olması kontrol edildi. Defter henüz gerçek Colab GPU oturumunda çalıştırılmadı. OpenCV'nin CUDA ile derlenmesi ve C++ CUDA çalıştırmasının başarısı doğrulanmış değildir.
