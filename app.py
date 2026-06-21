"""
FoodRec — Dinamik Doygunluk Temelli Hibrit Yemek Öneri Sistemi
=============================================================
Çalıştır : python app.py
Gerekli  : RAW_recipes.csv, RAW_interactions.csv (aynı klasörde)
Backend  : Flask  |  Port: 5000

DEĞİŞİKLİKLER (v3):
  - Alpha artık dinamik: etkileşim sayısına + seçili malzeme sayısına göre ayarlanıyor
  - Doygunluk cezası yumuşatıldı: 0.10 → 0.05, max 0.30 → 0.15
  - Ceza sadece s_base < 0.65 olan tariflere uygulanıyor (yüksek kaliteli öneriler korunuyor)
  - evaluate.py'den gelen best_alpha production'a taşındı
"""

import ast
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

app = Flask(__name__)
CORS(app)

# 1. VERİ YÜKLEMESİ  (sunucu açılışında bir kez çalışır)
try:
    print("-> Kaggle veri seti yükleniyor...")
    df_recipes      = pd.read_csv("RAW_recipes.csv")
    df_interactions = pd.read_csv("RAW_interactions.csv")

    print("-> Tarif malzeme haritası...")
    recipe_ingredients = {}
    for _, row in df_recipes.iterrows():
        try:
            ings = ast.literal_eval(row["ingredients"])
            recipe_ingredients[int(row["id"])] = [i.lower().strip() for i in ings]
        except Exception:
            recipe_ingredients[int(row["id"])] = []

    print("-> Kullanıcı profilleri...")
    user_liked   = defaultdict(set)
    user_ratings = {}
    for _, row in df_interactions[df_interactions["rating"] >= 4].iterrows():
        uid = int(row["user_id"])
        rid = int(row["recipe_id"])
        user_liked[uid].add(rid)
        user_ratings[(uid, rid)] = (float(row["rating"]) - 1) / 4.0

    print("-> Malzeme-kullanıcı indeksi...")
    ingredient_to_users = defaultdict(set)
    for uid, rids in user_liked.items():
        for rid in rids:
            for ing in recipe_ingredients.get(rid, []):
                ingredient_to_users[ing].add(uid)

    print("-> CF taban puanları...")
    cf_grouped    = df_interactions.groupby("recipe_id")["rating"].mean()
    cf_normalized = (cf_grouped - 1) / 4
    cf_dict       = cf_normalized.fillna(0.5).to_dict()
    df_recipes["base_cf_score"] = df_recipes["id"].map(cf_dict).fillna(0.5)

    print("-> Kullanıcı etkileşim sayıları...")
    user_interaction_counts = df_interactions.groupby("user_id").size().to_dict()

    print("-> Global TF-IDF matrisi...")
    df_recipes["clean_ing"] = df_recipes["ingredients"].fillna("").apply(
        lambda x: x.replace("[", "").replace("]", "").replace("'", "").replace(",", " ")
    )
    global_vectorizer   = TfidfVectorizer(max_features=8000)
    global_tfidf_matrix = global_vectorizer.fit_transform(df_recipes["clean_ing"])
    df_recipes          = df_recipes.reset_index(drop=True)

    # ── SVD + Item-based kNN
    print("-> SVD: kullanıcı-tarif matrisi oluşturuluyor...")
    all_recipe_ids   = df_recipes["id"].tolist()
    recipe_id_to_idx = {int(rid): i for i, rid in enumerate(all_recipe_ids)}
    all_user_ids     = list(user_liked.keys())
    user_id_to_idx   = {uid: i for i, uid in enumerate(all_user_ids)}

    rows_svd, cols_svd, vals_svd = [], [], []
    for uid, rids in user_liked.items():
        ui = user_id_to_idx[uid]
        for rid in rids:
            if rid in recipe_id_to_idx:
                rows_svd.append(ui)
                cols_svd.append(recipe_id_to_idx[rid])
                vals_svd.append(user_ratings.get((uid, rid), 1.0))

    n_users  = len(all_user_ids)
    n_items  = len(all_recipe_ids)
    R_sparse = csr_matrix((vals_svd, (rows_svd, cols_svd)), shape=(n_users, n_items))

    print("-> SVD ayrıştırması (k=50)...")
    k_factors    = min(50, min(n_users, n_items) - 1)
    U, sigma, Vt = svds(R_sparse.astype(float), k=k_factors)
    sigma_diag   = np.diag(sigma)

    item_factors      = np.dot(sigma_diag, Vt).T   # (n_items, k)
    norms             = np.linalg.norm(item_factors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    item_factors_norm = item_factors / norms        # cosine için normalize

    print("-> Sistem hazir!\n")

except FileNotFoundError:
    print("[HATA] CSV dosyalari bulunamadi.")
    df_recipes              = pd.DataFrame()
    recipe_ingredients      = {}
    user_liked              = defaultdict(set)
    user_ratings            = {}
    ingredient_to_users     = defaultdict(set)
    cf_dict                 = {}
    global_vectorizer       = None
    global_tfidf_matrix     = None
    user_interaction_counts = {}
    item_factors_norm       = None
    recipe_id_to_idx        = {}
    user_id_to_idx          = {}
    all_recipe_ids          = []


# MEVSIM → MALZEMESİ EŞLEMESİ
SEASON_INGREDIENTS = {
    "summer":  ["tomato", "zucchini", "basil", "cucumber", "corn",
                "peach", "watermelon", "eggplant", "pepper", "mint"],
    "winter":  ["potato", "carrot", "cabbage", "lentil", "squash",
                "parsnip", "turnip", "kale", "leek", "sweet potato"],
    "spring":  ["asparagus", "pea", "artichoke", "radish", "spinach",
                "lettuce", "fennel", "rhubarb", "leek", "arugula"],
    "autumn":  ["pumpkin", "apple", "pear", "mushroom", "butternut",
                "beet", "cranberry", "sage", "thyme", "chestnut"],
}


def season_score(ingredients_str: str, season: str) -> float:
    season_key  = season.lower().strip()
    season_ings = SEASON_INGREDIENTS.get(season_key, [])
    if not season_ings:
        return 0.0
    ing_lower = ingredients_str.lower()
    matches   = sum(1 for s in season_ings if s in ing_lower)
    return matches / len(season_ings)


# 2. CF FONKSİYONLARI
def compute_svd_knn_scores(user_id, liked_ings, candidate_ids):
    """
    Item-based kNN via SVD latent space.
    Kullanıcının beğendiği tariflerin latent vektörlerinin ortalamasını
    aday tariflerin vektörleriyle karşılaştırır.
    """
    if item_factors_norm is None:
        return {}

    liked_rids = []
    if user_id and user_id in user_liked:
        liked_rids = list(user_liked[user_id])
    elif liked_ings:
        for ing in liked_ings[:5]:
            for uid in list(ingredient_to_users.get(ing.lower().strip(), set()))[:20]:
                liked_rids.extend(list(user_liked[uid])[:3])
        liked_rids = list(set(liked_rids))[:30]

    if not liked_rids:
        return {}

    liked_vecs = []
    for rid in liked_rids:
        idx = recipe_id_to_idx.get(int(rid))
        if idx is not None:
            liked_vecs.append(item_factors_norm[idx])

    if not liked_vecs:
        return {}

    user_profile = np.mean(liked_vecs, axis=0)
    norm = np.linalg.norm(user_profile)
    if norm < 1e-9:
        return {}
    user_profile_norm = user_profile / norm

    scores = {}
    for rid in candidate_ids:
        idx = recipe_id_to_idx.get(int(rid))
        if idx is not None:
            sim = float(np.dot(user_profile_norm, item_factors_norm[idx]))
            scores[rid] = max(0.0, sim)

    return scores


def compute_cf_scores(likes, candidate_ids, user_id=None):
    """
    Hibrit CF Skoru (v3 — SVD ağırlığı artırıldı):
      - Neighborhood CF : malzeme tabanlı benzer kullanıcı oyları
      - SVD + item-based kNN : latent faktör uzayında tarif benzerliği
      - Genel popularite
    Final = 0.25 × neighborhood + 0.60 × SVD_kNN + 0.15 × popularite
    (önceki: 0.35 / 0.45 / 0.20)
    """
    popularity = {rid: cf_dict.get(rid, 0.5) for rid in candidate_ids}

    # Neighborhood CF 
    neighborhood_scores = {}
    if likes and user_liked:
        similar_users = set()
        for like in likes:
            like_lower = like.lower().strip()
            if like_lower in ingredient_to_users:
                similar_users.update(ingredient_to_users[like_lower])
            else:
                for ing_key in ingredient_to_users:
                    if like_lower in ing_key or ing_key in like_lower:
                        similar_users.update(ingredient_to_users[ing_key])

        if similar_users:
            vote_count = Counter()
            for uid in similar_users:
                for rid in user_liked[uid]:
                    vote_count[rid] += 1
            max_votes = max(vote_count.values()) if vote_count else 1
            neighborhood_scores = {
                rid: vote_count.get(rid, 0) / max_votes
                for rid in candidate_ids
            }

    # SVD + item-based kNN 
    svd_scores = compute_svd_knn_scores(user_id, likes, candidate_ids)

    # Final hibrit CF (v3 ağırlıkları)
    scores = {}
    for rid in candidate_ids:
        n_score   = neighborhood_scores.get(rid, 0.0)
        s_score   = svd_scores.get(rid, 0.0)
        pop_score = popularity.get(rid, 0.5)

        if svd_scores and neighborhood_scores:
            scores[rid] = 0.25 * n_score + 0.60 * s_score + 0.15 * pop_score
        elif svd_scores:
            scores[rid] = 0.70 * s_score + 0.30 * pop_score
        elif neighborhood_scores:
            scores[rid] = 0.65 * n_score + 0.35 * pop_score
        else:
            scores[rid] = pop_score

    return scores

# 3. DİNAMİK ALPHA — etkileşim + malzeme sayısına göre CB/CF dengesi
def get_dynamic_alpha(user_id=None, n_likes=0):
    """
    Alpha = CB ağırlığı.  Düşük alpha → CF baskın → precision ↑

    user_id varsa (kayıtlı kullanıcı):
      - Çok etkileşim → CF güvenilir → alpha düşük
      - Az etkileşim  → CB daha güvenilir → alpha yüksek
    user_id yoksa (misafir):
      - Sadece seçilen malzeme sayısına göre ayarlanır
    """
    if user_id is not None:
        n = user_interaction_counts.get(int(user_id), 0)
        if n >= 200:
            base_alpha = 0.10   # çok aktif kullanıcı → CF'ye güven
        elif n >= 50:
            base_alpha = 0.15
        elif n >= 10:
            base_alpha = 0.20
        else:
            base_alpha = 0.25   # yeni kullanıcı → biraz CB de olsun
    else:
        # Misafir: seçilen malzeme sayısı ne kadar fazlaysa CB o kadar güvenilir
        base_alpha = min(0.50, 0.30 + n_likes * 0.02)

    return base_alpha


# 4. ADAPTIVE THRESHOLD — kişisel doygunluk eşiği
def get_adaptive_threshold(user_id=None):
    """
    Kullanıcının toplam etkileşim sayısına göre maksimum ceza sınırı.
    v3: değerler yarıya indirildi — ceza daha hafif, precision korunuyor.
      <= 10  : 0.15  (misafir / yeni kullanıcı)
      <= 50  : 0.12
      <= 200 : 0.10
      200+   : 0.08  (çok aktif)
    """
    if user_id is None:
        return 0.15
    n = user_interaction_counts.get(int(user_id), 0)
    if   n <= 10:  return 0.15
    elif n <= 50:  return 0.12
    elif n <= 200: return 0.10
    else:          return 0.08


# 5. KULLANICI PROFİLİ
def get_user_profile(user_id):
    uid              = int(user_id)
    liked_recipe_ids = user_liked.get(uid, set())
    if not liked_recipe_ids:
        return [], 0
    ing_counter = Counter()
    for rid in liked_recipe_ids:
        for ing in recipe_ingredients.get(rid, []):
            ing_counter[ing] += 1
    return [ing for ing, _ in ing_counter.most_common(10)], len(liked_recipe_ids)


# 6. DOYGUNLUK ANALİZİ
def analyze_saturation(history):
    """
    Zaman ağırlıklı sayım:
      < 1 gün: 1.00  |  1-2 gün: 0.70  |  2-3 gün: 0.40  |  3+ gün: 0.15

    v3 ceza: max(0, w - 2.0) × 0.05  — ilk 2 seçimde ceza yok, 3.'den başlar.
    Maksimum ceza: %15  (önceki: %30)
    """
    now_ms     = datetime.now(timezone.utc).timestamp() * 1000
    one_day_ms = 86_400_000

    weighted_counts = Counter()
    for item in history:
        name     = item.get("name", "").lower().strip()
        ts       = item.get("ts", now_ms)
        days_ago = (now_ms - ts) / one_day_ms
        if   days_ago < 1: weight = 1.00
        elif days_ago < 2: weight = 0.70
        elif days_ago < 3: weight = 0.40
        else:              weight = 0.15
        weighted_counts[name] += weight

    # v3: katsayı 0.10 → 0.05, max 0.30 → 0.15
    penalty_map = {
        ing: min(0.15, max(0.0, w - 2.0) * 0.05)
        for ing, w in weighted_counts.items()
    }

    summary = []
    for ing, w in sorted(weighted_counts.items(), key=lambda x: -x[1]):
        if w >= 0.4:
            times       = int(round(w / 0.7)) if w < 2 else int(round(w))
            penalty_pct = int(penalty_map[ing] * 100)
            summary.append({
                "ingredient":   ing,
                "approx_times": max(1, times),
                "penalty_pct":  penalty_pct,
                "heavy":        w >= 2.0,
            })

    return penalty_map, summary


# 7. HİBRİT ÖNERİ
def get_recommendations(likes, dislikes, history, meal_context,
                        season_context, top_k, max_minutes=None,
                        user_id=None, seen_ids=None):
    """
    S_base  = alpha × S_CB + (1 - alpha) × S_CF
    S_final = S_base × max(0.85, 1 - penalty)   ← ceza SADECE s_base < 0.65 ise uygulanır

    alpha (v3): get_dynamic_alpha() ile etkileşim + malzeme sayısına göre belirlenir.
    """
    if df_recipes.empty or global_vectorizer is None:
        return [], []

    seen_ids             = set(seen_ids or [])
    penalty_map, sat_sum = analyze_saturation(history)
    adaptive_thr         = get_adaptive_threshold(user_id)

    # 1. Öğün Filtrelemesi 
    filtered_df = df_recipes[
        df_recipes["tags"].str.contains(meal_context.lower(), case=False, na=False)
    ].copy()

    #  Mevsim Filtresi 
    season_key = season_context.lower().strip() if season_context else ""
    if season_key:
        tmp = filtered_df[
            filtered_df["tags"].str.contains(season_key, case=False, na=False)
        ]
        if len(tmp) >= 20:
            filtered_df = tmp
        else:
            filtered_df["season_score"] = filtered_df["ingredients"].apply(
                lambda x: season_score(str(x), season_key)
            )

    if max_minutes and int(max_minutes) < 999:
        filtered_df = filtered_df[filtered_df["minutes"] <= int(max_minutes)]

    if seen_ids:
        filtered_df = filtered_df[~filtered_df["id"].isin(seen_ids)]

    if len(filtered_df) < 20:
        filtered_df = df_recipes.copy()
        if seen_ids:
            filtered_df = filtered_df[~filtered_df["id"].isin(seen_ids)]
        if max_minutes and int(max_minutes) < 999:
            filtered_df = filtered_df[filtered_df["minutes"] <= int(max_minutes)]

    # 2. Dislike Filtresi 
    import re
    if dislikes:
        for d in dislikes:
            pattern = r'\b' + re.escape(d.lower().strip()) + r'\b'
            filtered_df = filtered_df[
                ~filtered_df["ingredients"].str.contains(pattern, case=False, na=False, regex=True)
            ]

    if filtered_df.empty:
        return [], sat_sum

    #  3. CB Skoru 
    user_str = " ".join([l.lower() for l in likes]) if likes else None

    pos_map     = {idx: pos for pos, idx in enumerate(df_recipes.index)}
    valid_index = [idx for idx in filtered_df.index if idx in pos_map]
    if not valid_index:
        return [], sat_sum

    filtered_df   = filtered_df.loc[valid_index].copy()
    row_positions = [pos_map[idx] for idx in valid_index]

    if user_str:
        user_vec  = global_vectorizer.transform([user_str])
        sub_tfidf = global_tfidf_matrix[row_positions]
        cb_arr    = cosine_similarity(user_vec, sub_tfidf).flatten()
    else:
        cb_arr = np.zeros(len(row_positions))

    if "season_score" in filtered_df.columns:
        season_arr = filtered_df["season_score"].values
        cb_arr     = cb_arr + 0.20 * season_arr
        cb_arr     = np.clip(cb_arr, 0.0, 1.0)

    filtered_df["cb_score"] = cb_arr

    # 4. CF Skoru
    cf_map = compute_cf_scores(likes, filtered_df["id"].tolist(), user_id=user_id)

    #5. Dinamik Alpha (v3) 
    alpha = get_dynamic_alpha(user_id=user_id, n_likes=len(likes))

    #  6. Final Skor 
    recommendations = []
    for _, row in filtered_df.iterrows():
        s_cb   = float(row["cb_score"])
        s_cf   = cf_map.get(int(row["id"]), 0.5)
        s_base = alpha * s_cb + (1 - alpha) * s_cf

        ing_lower     = str(row["ingredients"]).lower()
        total_penalty = 0.0
        matched_ings  = []
        for ing, pen in penalty_map.items():
            if ing in ing_lower:
                total_penalty += pen
                matched_ings.append(ing)

        # v3: Ceza SADECE düşük skorlu tariflere uygulanıyor
        # Yüksek kaliteli önerilerin skoru bozulmuyor → precision korunuyor
        if s_base < 0.65:
            penalty_factor = max(0.85, 1.0 - min(total_penalty, adaptive_thr))
        else:
            penalty_factor = 1.0

        s_final = s_base * penalty_factor

        try:
            display_ings = ast.literal_eval(row["ingredients"])
        except (ValueError, SyntaxError):
            display_ings = [str(row["ingredients"])]

        try:
            steps = ast.literal_eval(row["steps"])
        except (ValueError, SyntaxError):
            steps = [str(row.get("steps", ""))]

        recommendations.append({
            "id":           int(row["id"]),
            "name":         str(row["name"]).title(),
            "ingredients":  display_ings[:8],
            "minutes":      int(row["minutes"]),
            "steps":        steps,
            "meal_type":    meal_context,
            "cb_score":     round(s_cb, 3),
            "cf_score":     round(float(s_cf), 3),
            "penalty":      round(total_penalty, 3),
            "penalized_by": matched_ings,
            "final_score":  round(s_final, 3),
        })

    #  7. Sıralama + Softmax Seçim
    recommendations.sort(key=lambda x: x["final_score"], reverse=True)
    pool   = recommendations[:30]
    scores = np.array([r["final_score"] for r in pool])
    scores = scores - scores.max()
    probs  = np.exp(scores * 3)
    probs  = probs / probs.sum()

    idx      = np.random.choice(len(pool), size=min(top_k, len(pool)), replace=False, p=probs)
    selected = [pool[i] for i in sorted(idx)]
    return selected, sat_sum


# 8. API ENDPOINT'LERİ
@app.route("/api/users", methods=["GET"])
def users_endpoint():
    try:
        active  = sorted([int(uid) for uid, rids in user_liked.items() if len(rids) >= 5])
        sampled = active[:25]
        return jsonify({"success": True, "users": sampled}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_endpoint():
    loaded = not df_recipes.empty and global_vectorizer is not None
    return jsonify({
        "status":         "ok" if loaded else "degraded",
        "recipes_loaded": int(len(df_recipes)) if loaded else 0,
        "users_loaded":   int(len(user_liked)) if loaded else 0,
        "svd_ready":      item_factors_norm is not None,
    }), 200 if loaded else 503


@app.route("/api/user_profile/<int:user_id>", methods=["GET"])
def user_profile_endpoint(user_id):
    try:
        top_ings, liked_count = get_user_profile(user_id)
        return jsonify({
            "success":            True,
            "user_id":            user_id,
            "top_ingredients":    top_ings,
            "liked_count":        liked_count,
            "n_interactions":     user_interaction_counts.get(user_id, 0),
            "adaptive_threshold": get_adaptive_threshold(user_id),
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommend", methods=["POST"])
def recommend_endpoint():
    try:
        data = request.get_json() or {}
        results, saturation = get_recommendations(
            likes          = data.get("likes", []),
            dislikes       = data.get("dislikes", []),
            history        = data.get("history", []),
            meal_context   = data.get("meal", "Lunch"),
            season_context = data.get("season", "Summer"),
            top_k          = int(data.get("topk", 5)),
            max_minutes    = data.get("max_minutes", None),
            user_id        = data.get("user_id", None),
            seen_ids       = data.get("seen_ids", []),
        )
        return jsonify({
            "success":         True,
            "recommendations": results,
            "saturation":      saturation,
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)