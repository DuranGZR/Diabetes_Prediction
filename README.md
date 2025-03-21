# 🏥 Diabetes Prediction (Diyabet Tahmini)

Bu proje, **makine öğrenimi (ML) algoritmalarını kullanarak diyabet hastalığını tahmin etmeyi** amaçlayan bir sistem geliştirmektedir. Projede, çeşitli **veri analizi, ön işleme ve modelleme** adımları uygulanarak farklı makine öğrenimi algoritmalarının performansı karşılaştırılmıştır.

Diyabet, dünya genelinde milyonlarca insanı etkileyen yaygın bir hastalıktır. Erken teşhis, yaşam tarzı değişiklikleri ve medikal müdahaleler sayesinde diyabetin olumsuz etkileri en aza indirilebilir. Bu proje, **Pima Indians Diabetes Dataset** kullanarak, hastalık tahmini yapabilecek bir makine öğrenimi modeli geliştirmeyi hedeflemektedir.

---

## 📌 İçindekiler

- [🎯 Projenin Amacı](#🎯-projenin-amacı)
- [📂 Veri Seti](#📂-veri-seti)
- [⚙️ Kullanılan Teknolojiler](#⚙️-kullanılan-teknolojiler)
- [🛠️ Veri Ön İşleme](#🛠️-veri-ön-işleme)
- [📊 Modelleme](#📊-modelleme)
- [📈 Model Değerlendirme](#📈-model-değerlendirme)
- [🏗️ Proje Yapısı](#🏗️-proje-yapısı)
- [🚀 Nasıl Kullanılır?](#🚀-nasıl-kullanılır)
- [🔮 Gelecekteki Geliştirmeler](#🔮-gelecekteki-geliştirmeler)
- [👨‍💻 Katkıda Bulunanlar](#👨‍💻-katkıda-bulunanlar)

---

## 🎯 Projenin Amacı

Bu projenin temel amacı, **makine öğrenimi yöntemlerini kullanarak bireylerin diyabet hastası olup olmadığını tahmin eden bir model oluşturmaktır**. Bunun için aşağıdaki adımlar uygulanmıştır:

✅ **Veri keşfi ve analiz**  
✅ **Eksik ve hatalı verilerin tespiti ve düzeltilmesi**  
✅ **Özellik mühendisliği ve veri dönüşümleri**  
✅ **Farklı ML modellerinin eğitilmesi ve karşılaştırılması**  
✅ **En iyi modelin seçilmesi ve tahminler yapılması**

---

## 📂 Veri Seti

Bu proje, **Pima Indians Diabetes Dataset** adlı veri setini kullanmaktadır. Veri seti, **768 gözlem ve 8 bağımsız değişkenden** oluşmaktadır.

| Değişken | Açıklama |
|----------|-----------|
| **Pregnancies** | Gebelik sayısı |
| **Glucose** | Glikoz seviyesi (mg/dL) |
| **BloodPressure** | Kan basıncı (mm Hg) |
| **SkinThickness** | Cilt kalınlığı (mm) |
| **Insulin** | İnsülin seviyesi (mu U/ml) |
| **BMI** | Vücut kitle indeksi (kg/m²) |
| **DiabetesPedigreeFunction** | Genetik risk faktörü |
| **Age** | Yaş (yıl) |
| **Outcome** | Diyabet teşhisi (1: Pozitif, 0: Negatif) |

📂 Veri seti **datasets/** klasörü içinde **diabetes.csv** dosyasında bulunmaktadır.

---

## ⚙️ Kullanılan Teknolojiler

Bu projede kullanılan teknolojiler ve kütüphaneler:

- 🐍 **Python 3.9+**
- 📊 **Pandas, NumPy** (Veri manipülasyonu ve analizi)
- 📈 **Matplotlib, Seaborn** (Görselleştirme)
- 🔍 **Scikit-learn** (Makine öğrenimi modelleri)


---

## 🛠️ Veri Ön İşleme

1️⃣ **Eksik Verilerin Analizi**  
- Eksik değerler tespit edildi ve uygun yöntemlerle dolduruldu.

2️⃣ **Ölçeklendirme ve Dönüşümler**  
- **MinMaxScaler** ile veriler ölçeklendirildi.
- Kategorik değişkenler **One-Hot Encoding** ile kodlandı.

3️⃣ **Özellik Seçimi ve Mühendisliği**  
- Korelasyon analizi yapıldı.
- Gereksiz ve düşük etkili değişkenler elendi.

---

## 📊 Modelleme

Farklı makine öğrenimi algoritmaları eğitildi ve karşılaştırıldı:

🔹 **Lojistik Regresyon**  
🔹 **Karar Ağaçları (Decision Trees)**  
🔹 **Rastgele Orman (Random Forests)**  
🔹 **Destek Vektör Makineleri (SVM)**  
🔹 **K-En Yakın Komşu (KNN)**  

📌 En iyi model belirlenirken **doğruluk, hassasiyet, kesinlik ve F1 skoru** gibi metrikler kullanılmıştır.

---

## 📈 Model Değerlendirme

**Performans ölçümleri:**

- **Accuracy (Doğruluk)**
- **Precision (Kesinlik)**
- **Recall (Duyarlılık)**
- **F1 Score**
- **ROC-AUC Skoru**



---

## 🏗️ Proje Yapısı

📂 **datasets/** → Diyabet veri seti dosyası  
📂 **notebooks/** → Veri analizi ve modelleme için Jupyter Notebook dosyaları  
📂 **models/** → Eğitilmiş modellerin saklandığı dosyalar  
📜 **config.py** → Model yapılandırma ayarları  
📜 **diabetes_pipeline.py** → Veri işleme ve modelleme pipeline’ı  
📜 **diabetes_prediction.py** → Eğitilmiş model ile tahmin yapma kodu  




