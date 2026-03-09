# 🏠 Yapay Zeka Final Projesi: Ev Fiyat Tahmin Modeli

Bu proje, **Yapay Zeka ve Akıllı Sistemler** dersi final notu için hazırladığım, gerçek veriler üzerinden ev fiyatı tahmini yürüten bir çalışmadır.

### 📝 Proje Özeti
Projemde, **Kaggle** üzerinden aldığım ve Türkiye’deki çeşitli illere ait ev fiyatlarını içeren `home_price.csv` veri setini kullandım. Sistem, kullanıcıdan şu bilgileri alarak bir fiyat tahmini üretmektedir:
* 📍 **Şehir**
* 📐 **Metrekare**
* 🚪 **Oda Sayısı**

### 🧠 Algoritma ve Süreç
Bu projeyi geliştirirken **Lineer Regresyon (Linear Regression)** algoritmasını tercih ettim. 

> **Önemli Bir Not:** > Modeli eğitirken ve test ederken fark ettim ki, ev fiyatları çok fazla değişkene bağlı olduğu için Lineer Regresyon bu veri seti üzerinde biraz "basit" kaldı. Bu durum tahminlerin bazen hatalı veya beklenen değerlerin dışında çıkmasına neden olabiliyor. 
>
> Bu algoritma tercihinin eksikliğini fark ettiğimde teslim gününe çok az kalmıştı, bu yüzden daha karmaşık bir modele (Random Forest vb.) geçiş yapamadım. Ancak bu süreç bana, her algoritmanın her veri setine uygun olmayabileceğini bizzat yaşayarak öğretmiş oldu.

### 🛠️ Neler Öğrendim?
Hatalı tahminler olsa da projenin geliştirme sürecinde şu önemli adımları uyguladım:
1.  **Veri Temizleme:** Eksik verileri (NaN) mantıklı değerlerle doldurdum.
2.  **Veri Dönüştürme:** Şehir isimleri gibi kategorik verileri makinenin anlayacağı sayısal değerlere (One-Hot Encoding) çevirdim.
3.  **Ölçeklendirme:** Modelin daha iyi öğrenebilmesi için `StandardScaler` kullandım.
4.  **Matris İşlemleri:** Kullanıcıdan gelen veriyi `reshape` gibi yöntemlerle modelin beklediği matematiksel formata getirdim.

**Sonuç olarak;** bu proje benim için sadece bir fiyat tahmin aracı değil, bir verinin ham halinden tahmin aşamasına kadar geçtiği tüm zorlu yolları deneyimlediğim bir çalışma oldu. 😊 Projenin tüm teknik detayları ve mantığı, kod dosyaları içerisindeki yorum satırlarında açıklanmıştır.
