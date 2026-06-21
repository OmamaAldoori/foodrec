# 🍽️ FoodRec — Dynamic Satiety-Based Hybrid Food Recommendation System

> A hybrid food recommendation system built on the Food.com dataset, combining content-based filtering, collaborative filtering, and a dynamic satiety penalty mechanism.

---

## 📌 Overview

FoodRec recommends personalized recipes based on the user's preferred ingredients, meal time, season, and past interactions. A dynamic **satiety penalty** ensures the same ingredients are not repeatedly recommended, promoting variety over time.

---

## 🧠 How It Works

### 1. Content-Based Filtering (CB)
Vectorizes the user's liked ingredients using **TF-IDF** and finds the most similar recipes via **cosine similarity**.

```
S_CB = cosine_sim(liked_ingredients, recipe_ingredients)
```

### 2. Collaborative Filtering (CF)
A weighted combination of three components:

```
S_CF = 0.25 × Neighborhood + 0.60 × SVD_kNN + 0.15 × Popularity
```

- **Neighborhood CF**: Votes from users with similar ingredient preferences
- **SVD + Item-based kNN**: Recipe similarity in latent factor space (k=50)
- **Popularity**: Global average rating score

### 3. Hybrid Score & Satiety Penalty
Alpha is determined dynamically based on the user's interaction count and number of selected ingredients.

```
S_base  = α × S_CB + (1 - α) × S_CF
S_final = S_base × (1 - Penalty)   ← applied only when S_base < 0.65
```

---

## 📁 Project Structure

```
foodrec/
├── app.py        # Flask backend — recommendation engine and API endpoints
├── index.html    # Frontend interface
├── style.css     # Styling
├── app.js        # Frontend logic (API calls, tag management, history)
└── README.md
```

> ⚠️ `RAW_recipes.csv` and `RAW_interactions.csv` are not included due to file size.  
> Download them from [Food.com on Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

---

## 🚀 Getting Started

### Requirements

```bash
pip install flask flask-cors pandas numpy scikit-learn scipy matplotlib
```

### Download the Dataset

Download `RAW_recipes.csv` and `RAW_interactions.csv` from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) and place them in the project folder.

### Start the Backend

```bash
python app.py
```

The server runs at `http://127.0.0.1:5000`.

### Open the Frontend

Open `index.html` in your browser, or serve it with a local HTTP server:

```bash
python -m http.server 8080
```

Then visit `http://localhost:8080`.

---

## 🗂️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server and data status |
| `/api/users` | GET | List of active users |
| `/api/user_profile/<id>` | GET | User profile and ingredient preferences |
| `/api/recommend` | POST | Personalized recipe recommendations |

### Example `/api/recommend` Request

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

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask, Flask-CORS |
| ML / Recommendation | scikit-learn (TF-IDF, cosine similarity), scipy (SVD, sparse matrix) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Dataset | Food.com (Kaggle) — 180K+ recipes, 700K+ ratings |

---

## 📈 Model Version (v3)

- SVD weight increased: `0.45 → 0.60`
- Satiety penalty coefficient softened: `0.10 → 0.05`, max `0.30 → 0.15`
- Penalty applied only to recipes where `S_base < 0.65`
- Alpha is now dynamic: adjusted based on user interaction count and number of liked ingredients
