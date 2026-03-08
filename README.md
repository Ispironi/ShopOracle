# 🛒 ShopOracle Alışveriş Tahmini(Gemini + Deep Learning)

Bu proje, kullanıcıların doğal dille yazdığı alışveriş senaryolarını analiz ederek, müşterinin hangi kategoride alışveriş yapacağını tahmin eden hibrit bir makine öğrenmesi ve yapay zeka uygulamasıdır. 

Projeyle hem yapılandırılmamış verileri (serbest metin) anlamlandırmak hem de derin öğrenme (Deep Learning) ile sınıflandırma ve regresyon analizleri yapmak hedeflenmiştir.

## ✨ Özellikler

* **Doğal Dil İşleme (LLM):** Kullanıcıdan alınan serbest metin (örn: *"30 yaşında bir erkek, Cevahir AVM'de hafta sonu kredi kartıyla 2500 TL'lik alışveriş yapacak."*), **Google Gemini 2.5 Flash** API'si kullanılarak analiz edilir ve modelin anlayabileceği yapılandırılmış JSON formatına dönüştürülür.
* **Kategori Tahmini (Sınıflandırma):** Gemini'den elde edilen veriler, önceden eğitilmiş bir **Derin Sinir Ağı (MLP)** modeline verilerek ürün kategorisi (Giyim, Teknoloji, Kozmetik vb.) %99+ doğrulukla tahmin edilir.
* **Modern ve Dinamik Arayüz:** Kullanıcıların tahminleri canlı olarak test edebilmesi için Flask tabanlı, estetik bir web arayüzü (HTML/CSS/JS) sunulur.
* **Model Kayıt ve Entegrasyon:** Scikit-learn ile hazırlanan scaler/encoder objeleri ve Keras modeli dışa aktarılmış (`.h5` ve `.pkl`), web sunucusuna entegre edilmiştir.

## 🛠️ Kullanılan Teknolojiler

* **Yapay Zeka & Derin Öğrenme:** TensorFlow / Keras, Google GenAI (Gemini API)
* **Veri Bilimi:** Pandas, NumPy, Scikit-learn, Joblib
* **Backend:** Python, Flask
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

## 🚀 Kurulum ve Kullanım

### 1. Gereksinimleri Yükleyin
Projeyi yerel makinenizde çalıştırmak için terminalden aşağıdaki kütüphaneleri kurun:
`bash
pip install flask numpy scikit-learn tensorflow google-genai joblib
`

### 2. Gemini API Key Ayarı (Zorunlu)
Uygulamanın metin analizi yapabilmesi için ücretsiz bir Google Gemini API anahtarına ihtiyacınız var. 
1. [Google AI Studio](https://aistudio.google.com/) adresine gidin ve ücretsiz bir API Key oluşturun.
2. `app.py` dosyasını bir metin editörü ile açın.
3. Dosyanın içindeki `GEMINI_API_KEY = "BURAYA_KENDI_API_KEYINIZI_YAZIN"` satırını bulun ve tırnak işaretlerini silmeden kendi anahtarınızı yapıştırın.

### 3. Klasör Düzeni:
Uygulamanın sorunsuz çalışması için Flask'in aradığı dosya yapısını şu şekilde kurmanız gerekir:

Senin_Proje_Klasorun/
├── app.py              # Flask sunucusu ve Gemini entegrasyonu.
├── model.h5            # Eğitilmiş Derin Öğrenme modeli.
├── encoders.pkl        # Kategorik veri kodlayıcıları (Scikit-learn).
├── scaler.pkl          # Sayısal veri ölçekleyicileri (Scikit-learn).
├── thresholds.pkl      # Fiyat segmentasyonu eşik değerleri.
└── templates/          # DİKKAT: Bu klasörü elinizle açmalısınız!.
    └── index.html      # Web arayüzü tasarımı bu klasörün içinde olmalı.

### 4. Sunucuyu Başlatın
Terminali (veya CMD/PowerShell) projenin bulunduğu dizinde açın ve Flask sunucusunu ayağa kaldırın:
`bash
python app.py
`
Terminalde `* Running on http://127.0.0.1:5000` yazısını gördükten sonra, tarayıcınızı açın ve **http://127.0.0.1:5000** adresine giderek uygulamayı kullanmaya başlayın!
