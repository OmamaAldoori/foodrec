# 🍽️ FoodRec — Dynamic Satiety-Based Hybrid Food Recommendation System
# 🍽️ FoodRec — Dinamik Doygunluk Temelli Hibrit Yemek Öneri Sistemi

> 🇬🇧 A hybrid food recommendation system built on the Food.com dataset, combining content-based filtering, collaborative filtering, and a dynamic satiety penalty mechanism.
>
> 🇹🇷 Food.com veri seti üzerinde içerik tabanlı filtreleme, işbirlikçi filtreleme ve dinamik doygunluk cezasını birleştiren hibrit bir yemek öneri sistemi.

---

## 📌 Overview / Genel Bakış

🇬🇧 FoodRec recommends personalized recipes based on the user's preferred ingredients, meal time, season, and past interactions. A dynamic **satiety penalty** ensures the same ingredients are not repeatedly recommended, promoting variety over time.

🇹🇷 FoodRec, kullanıcının sevdiği malzemelere, öğün zamanına, mevsimine ve geçmiş etkileşimlerine göre kişiselleştirilmiş tarif önerileri sunar. Dinamik **doygunluk cezası** sayesinde aynı malzemelerin tekrar önerilmesi engellenir.

---

## 🧠 How It Works / Nasıl Çalışır?

### 1. Content-Based Filtering / İçerik Tabanlı Filtreleme (CB)

🇬🇧 Vectorizes the user's liked ingredients using **TF-IDF** and finds the most similar recipes via **cosine similarity**.  
🇹🇷 Kullanıcının sevdiği malzemeleri **TF-IDF** ile vektörleştirir ve **kosinüs benzerliği** ile en yakın tarifleri bulur.

```
S_CB = cosine_sim(liked_ingredients, recipe_ingredients)
```

### 2. Collaborative Filtering / İşbirlikçi Filtreleme (CF)

🇬🇧 A weighted combination of three components:  
🇹🇷 Üç bileşenin ağırlıklı toplamından oluşur:

```
S_CF = 0.25 × Neighborhood + 0.60 × SVD_kNN + 0.15 × Popularity
```

- **Neighborhood CF**: 🇬🇧 Votes from users with similar ingredient preferences / 🇹🇷 Benzer malzeme tercihlerine sahip kullanıcıların oyları
- **SVD + Item-based kNN**: 🇬🇧 Recipe similarity in latent factor space (k=50) / 🇹🇷 Latent faktör uzayında tarif benzerliği (k=50)
- **Popularity**: 🇬🇧 Global average rating score / 🇹🇷 Genel ortalama puan

### 3. Hybrid Score & Satiety Penalty / Hibrit Skor & Doygunluk Cezası

🇬🇧 Alpha is determined dynamically based on the user's interaction count and number of selected ingredients.  
🇹🇷 Alpha, kullanıcı etkileşim sayısına ve seçili malzeme sayısına göre dinamik olarak belirlenir.

```
S_base  = α × S_CB + (1 - α) × S_CF
S_final = S_base × (1 - Penalty)   ← applied only when S_base < 0.65
```

---

## 📁 Project Structure / Dosya Yapısı

```
foodrec/
├── backend/
│   ├── app.py                    # Flask backend — recommendation engine & API
│   ├── RAW_recipes.csv           # ⚠️ Not included — download from Kaggle
│   └── RAW_interactions.csv      # ⚠️ Not included — download from Kaggle
├── index.html                    # Frontend interface / Arayüz
├── style.css                     # Styling / Stil
├── app.js                        # Frontend logic / Frontend mantığı
└── README.md
```

> ⚠️ 🇬🇧 `RAW_recipes.csv` and `RAW_interactions.csv` must be placed in the **same folder as `app.py`**. Download from [Food.com on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).
>
> ⚠️ 🇹🇷 `RAW_recipes.csv` ve `RAW_interactions.csv` dosyaları `app.py` ile **aynı klasörde** bulunmalıdır. [Kaggle'dan](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) indirebilirsiniz.

---

## 🚀 Getting Started / Kurulum

### Requirements / Gereksinimler

```bash
pip install flask flask-cors pandas numpy scikit-learn scipy matplotlib
```

### Start the Backend / Backend'i Başlat

```bash
python app.py
```

🇬🇧 Server runs at `http://127.0.0.1:5000`  
🇹🇷 Sunucu `http://127.0.0.1:5000` adresinde çalışır

### Open the Frontend / Frontend'i Aç

```bash
python -m http.server 8080
```

🇬🇧 Then visit `http://localhost:8080`  
🇹🇷 Ardından `http://localhost:8080` adresini ziyaret et

---

## 🗂️ API Endpoints

| Endpoint | Method | 🇬🇧 Description | 🇹🇷 Açıklama |
|----------|--------|----------------|--------------|
| `/api/health` | GET | Server and data status | Sunucu ve veri durumu |
| `/api/users` | GET | List of active users | Aktif kullanıcı listesi |
| `/api/user_profile/<id>` | GET | User profile and preferences | Kullanıcı profili |
| `/api/recommend` | POST | Personalized recommendations | Kişiselleştirilmiş öneriler |

### Example Request / Örnek İstek

```json
{
  "likes": ["chicken", "garlic", "tomato"],
  "dislikes": ["cream", "mayo"],
  "meal": "Dinner",
  "season": "Winter",
  "topk": 5,
  "max_minutes": 45,
  "history": ["chicken", "garlic"],
  "user_id": null
}
```

---

## 🛠️ Tech Stack / Teknolojiler

| Layer / Katman | Technology / Teknoloji |
|----------------|------------------------|
| Backend | Python, Flask, Flask-CORS |
| ML / Recommendation | scikit-learn (TF-IDF, cosine similarity), scipy (SVD, sparse matrix) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Dataset / Veri Seti | Food.com (Kaggle) — 180K+ recipes / tarif, 700K+ ratings / değerlendirme |

---

## 📈 Model Version / Model Versiyonu (v3)

- 🇬🇧 SVD weight increased: `0.45 → 0.60` / 🇹🇷 SVD ağırlığı artırıldı
- 🇬🇧 Satiety penalty softened: `0.10 → 0.05`, max `0.30 → 0.15` / 🇹🇷 Doygunluk cezası yumuşatıldı
- 🇬🇧 Penalty applied only when `S_base < 0.65` / 🇹🇷 Ceza yalnızca `S_base < 0.65` ise uygulanıyor
- 🇬🇧 Alpha is now dynamic / 🇹🇷 Alpha artık dinamik

---

## 👤 Developer / Geliştirici

**OmamaAldoori** · Food.com Dataset · Hybrid Recommendation System · 2025
