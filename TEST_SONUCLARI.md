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
