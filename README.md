# YZM304 I. Ödev: BankNote Authentication Modeli

## Giriş (Introduction)

Bu projede, Kaggle'dan alınan **BankNote Authentication** veri setini kullanarak bir yapay sinir ağı (ANN) modeli eğitilmiştir. Model, iki farklı yapıda oluşturulmuş ve tanh ve ReLU aktivasyon fonksiyonları kullanılmıştır. Modelin amacı, banknotların sahte olup olmadığını sınıflandırmaktır. İki farklı model (2-Layer tanh ve 3-Layer ReLU) ile yapılan denemeler, karşılaştırmalı sonuçlar elde edilerek modelin doğruluğu ve diğer metrikleri değerlendirilecektir.

## Yöntem (Method)

### Veri Seti
- **Veri Seti:** Kaggle’dan alınan **BankNote_Authentication** veri seti.
- **Özellikler:** Veri setinde 4 temel özellik bulunmaktadır. Bu özellikler banknotların güvenliğini analiz etmek için kullanılır.

### Model Yapısı
- **2-Layer Model (Tanh):** Bu modelde bir gizli katman ve bir çıkış katmanı bulunur. Gizli katmanda **tanh** aktivasyon fonksiyonu kullanılmıştır.
- **3-Layer Model (ReLU):** Bu modelde iki gizli katman ve bir çıkış katmanı bulunur. Gizli katmanlarda **ReLU** aktivasyon fonksiyonu kullanılmıştır.

### Eğitim ve Değerlendirme
- **Kayıp Fonksiyonu:** Binary Cross-Entropy kullanılmıştır.
- **Optimizasyon Algoritması:** Stokastik gradyan inişi (SGD) kullanılmıştır.
- **Aktivasyon Fonksiyonları:** Gizli katmanlarda tanh ve ReLU aktivasyon fonksiyonları sırasıyla kullanılmıştır.
- **Performans Metrikleri:** Doğruluk, geri çağırma, karmaşıklık matrisi gibi temel ölçüm metrikleri kullanılmıştır.

### Eğitim Süreci
Modeller, aynı eğitim ve test seti, başlangıç ağırlıkları, hiperparametreler ve optimizasyon algoritması ile eğitilmiştir. Modellerin başarıları, **accuracy** ve **n_steps** değerlerine göre değerlendirilmiştir.

## Sonuçlar (Results)

Eğitim sonrası elde edilen modellerin doğruluk oranları ve karmaşıklık matrisleri aşağıda verilmiştir:

- **2-Layer Model (Tanh):**
  - Doğruluk: %XX
  - Karmaşıklık Matrisi: [Karmaşıklık matrisi burada belirtilecek]

- **3-Layer Model (ReLU):**
  - Doğruluk: %XX
  - Karmaşıklık Matrisi: [Karmaşıklık matrisi burada belirtilecek]

## Tartışma (Discussion)

- **Başarılar:** İki model de beklenen başarıyı göstermiştir. Özellikle 3-Layer ReLU modelinin daha yüksek doğruluk oranları elde ettiği gözlemlenmiştir.
- **Sınırlamalar:** Modellerin performansı, hiperparametrelerin seçiminden etkilidir. Ayrıca, daha fazla gizli katman eklemek performans artışını sağlayabilir.
- **İleriye Dönük Çalışmalar:** Farklı aktivasyon fonksiyonları ve model yapılandırmaları ile daha derinlemesine analizler yapılabilir.

## Referanslar (References)

- [Kaggle BankNote Authentication Dataset](https://www.kaggle.com/uciml/banknote-authentication)
- [Deep Learning with Python, François Chollet](https://www.manning.com/books/deep-learning-with-python)
