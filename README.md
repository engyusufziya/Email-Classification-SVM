# Email Classification with SVM

📧 **Destek Vektör Makineleri (SVM) ile E-posta Sınıflandırması**  

Bu proje, bir e-postanın **Kişisel (0)** mı yoksa **İş (1)** e-postası mı olduğunu, gönderici formalitesi ve ilişki puanlarına dayalı olarak tahmin etmek için bir **Destek Vektör Makineleri (SVM)** modeli uygular. Bu proje SVM'nin çalışma mantığını anlamak açısından basit bir veriseti ile çalışıldı.

---

## 🚀 Proje Özellikleri

- **Model:** Destek Vektör Makineleri (SVM)  
- **Çekirdekler (Kernels):** Lineer (Linear) ve RBF (Radial Basis Function)  
- **Veri Seti:** `email_classification_svm.csv`  
- **Kütüphaneler:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`  

---

## 📊 Veri Seti ve Özellikler

Veri seti, **1000 gözlem** ve üç sütundan oluşmaktadır:  

| Özellik Adı                  | Açıklama                                                     |
|-------------------------------|-------------------------------------------------------------|
| `subject_formality_score`     | E-posta konusunun formalite puanı                            |
| `sender_relationship_score`   | Gönderici ile olan ilişkinin yakınlık puanı                 |
| `email_type` (Hedef)          | E-posta türü: `0` = Kişisel, `1` = İş E-postası            |

> Not: Özellikler, model performansını optimize etmek için muhtemelen önceden normalize edilmiştir.

---

## 🔍 Veri Görselleştirme (Separability)

Not defterindeki görselleştirmeler (pairplot ve scatter plot), veri noktalarının **iki sınıfa (0 ve 1) göre oldukça net ayrıldığını** göstermektedir. Bu yüksek ayrılabilirlik, SVM gibi sınırlayıcı hiperdüzlem tabanlı modellerin yüksek performans göstermesinin temel nedenidir.

---

## ⚙️ Metodoloji

1. **Veri Yükleme ve İnceleme**  
   - Eksik değerler ve genel dağılım kontrol edildi.

2. **Veri Bölme**  
   - %75 eğitim (`X_train`, `y_train`) ve %25 test (`X_test`, `y_test`) olarak ayrıldı (`random_state=42`).

3. **Model Eğitimi (Linear Kernel)**  
   - İlk olarak Lineer SVM modeli eğitildi.

4. **Model Eğitimi (RBF Kernel)**  
   - İkinci olarak RBF kernel ile SVM modeli eğitildi.

5. **Değerlendirme**  
   - Test verisi üzerinde `classification_report` ve `confusion_matrix` kullanıldı.

---

## ✅ Sonuçlar ve Performans

**Lineer SVM modeli**, test veri setinde neredeyse mükemmel sınıflandırma başarısı göstermiştir:

| Metrik                  | Sonuç (Lineer SVM) |
|-------------------------|------------------|
| Doğruluk (Accuracy)     | ~99-100%         |
| F1-Score (Ağırlıklı)   | Yüksek           |

> Model, yalnızca bir veya iki test örneğinde hata yapmıştır. Bu, veri setindeki sınıfların yüksek ayrılabilirliği göz önüne alındığında beklenen bir durumdur.

---

### 📝 RBF Model Değerlendirmesi

Not defterinde RBF kernel ile model eğitilmiş olmasına rağmen, değerlendirme aşamasında yanlışlıkla Lineer SVM modeli kullanılmıştır:

```python
# Hatalı Kod
y_pred = svc.predict(X_test)  # Doğru olan: rbf.predict(X_test)


