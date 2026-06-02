import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.vibes import VIBE_ANCHORS
from src.embeddings import create_playlist_embedding

model = SentenceTransformer("all-MiniLM-L6-v2")

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "key", "mode", "time_signature",
]

FEATURE_RANGES = {
    "danceability":     {"min": 0.0,   "max": 1.0},
    "energy":           {"min": 0.0,   "max": 1.0},
    "loudness":         {"min": -60.0, "max": 0.0},
    "speechiness":      {"min": 0.0,   "max": 1.0},
    "acousticness":     {"min": 0.0,   "max": 1.0},
    "instrumentalness": {"min": 0.0,   "max": 1.0},
    "liveness":         {"min": 0.0,   "max": 1.0},
    "valence":          {"min": 0.0,   "max": 1.0},
    "tempo":            {"min": 50.0,  "max": 220.0},
    "mode":             {"min": 0.0,   "max": 1.0},
    "time_signature":   {"min": 0.0,   "max": 5.0},
}

print("Encoding vibe anchors...")
_anchor_descriptions = [a["description"] for a in VIBE_ANCHORS]
_anchor_embeddings = model.encode(_anchor_descriptions, show_progress_bar=False)
print("Vibe anchors ready.")


def interpret_playlist(playlist_name: str, top_n_anchors: int = 3,
                       dominance_margin: float = 0.025) -> dict:

    playlist_vec = model.encode([playlist_name])
    similarities = cosine_similarity(playlist_vec, _anchor_embeddings)[0]

    sorted_idx = similarities.argsort()[::-1]
    top_score = similarities[sorted_idx[0]]
    second_score = similarities[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0

    effective_n = 1 if (top_score - second_score) >= dominance_margin else top_n_anchors

    top_indices = sorted_idx[:effective_n]
    top_scores = similarities[top_indices]

    weights = np.exp(top_scores) / np.exp(top_scores).sum()

    blended_features = {}
    for feature in AUDIO_FEATURES:
        blended_features[feature] = sum(
            weights[i] * VIBE_ANCHORS[top_indices[i]]["features"][feature]
            for i in range(effective_n)
        )

    genre_scores = {}
    for i in range(effective_n):
        anchor = VIBE_ANCHORS[top_indices[i]]
        for genre in anchor["genres"]:
            genre_scores[genre] = genre_scores.get(genre, 0) + weights[i]

    allowed_genres = [g for g, s in genre_scores.items() if s > 0.1]
    matched_anchors = [VIBE_ANCHORS[top_indices[i]]["description"] for i in range(effective_n)]

    year_range = VIBE_ANCHORS[top_indices[0]].get("year_range", None)

    return {
        "features": blended_features,
        "genres": allowed_genres,
        "matched_anchors": matched_anchors,
        "anchor_weights": weights.tolist(),
        "year_range": year_range,
    }


def get_relevant_clusters(target_features: dict, scaler, kmeans, top_n: int = 3) -> set:
    feature_vec = np.array([[target_features[f] for f in AUDIO_FEATURES]])
    normalized = scaler.transform(feature_vec)
    distances = np.linalg.norm(kmeans.cluster_centers_ - normalized, axis=1)
    return set(distances.argsort()[:top_n].tolist())


def score_songs_by_year(df, year_range: tuple) -> np.ndarray:
    min_y, max_y = year_range
    years = df["year"].values.astype(float)
    scores = np.ones(len(df))
    below = years < min_y
    scores[below] = np.maximum(0.0, 1.0 - (min_y - years[below]) / 10.0)
    above = years > max_y
    scores[above] = np.maximum(0.0, 1.0 - (years[above] - max_y) / 10.0)
    return scores


def score_songs_by_audio(df, target_features: dict, allowed_genres: list,
                         relevant_clusters: set = None, year_range: tuple = None) -> np.ndarray:
    feature_scores = np.zeros(len(df))

    for feature, target_val in target_features.items():
        if feature not in df.columns:
            continue

        if feature == "key":
            diff = np.abs(df[feature].values.astype(float) - target_val)
            circular_diff = np.minimum(diff, 12.0 - diff) / 6.0
            closeness = 1.0 - circular_diff
        elif feature not in FEATURE_RANGES:
            continue
        else:
            feat_min = FEATURE_RANGES[feature]["min"]
            feat_range = FEATURE_RANGES[feature]["max"] - feat_min
            normalized_vals = (df[feature].values - feat_min) / feat_range
            normalized_target = (target_val - feat_min) / feat_range
            closeness = 1.0 - np.abs(normalized_vals - normalized_target)

        feature_scores += closeness

    feature_scores /= len(target_features)

    keywords = set()
    for g in allowed_genres:
        for part in g.replace("-", " ").split():
            if len(part) >= 3:
                keywords.add(part)

    if keywords:
        pattern = "|".join(keywords)
        genre_mask = df["genre"].str.contains(pattern, case=False, na=False).values.astype(float)
    else:
        genre_mask = df["genre"].isin(allowed_genres).values.astype(float)

    has_clusters = relevant_clusters is not None and "cluster" in df.columns
    has_year = year_range is not None and "year" in df.columns

    cluster_mask = df["cluster"].isin(relevant_clusters).values.astype(float) if has_clusters else None
    year_scores = score_songs_by_year(df, year_range) if has_year else None

    if has_clusters and has_year:
        return (0.45 * feature_scores) + (0.20 * genre_mask) + (0.17 * cluster_mask) + (0.18 * year_scores)
    elif has_clusters:
        return (0.55 * feature_scores) + (0.25 * genre_mask) + (0.20 * cluster_mask)
    elif has_year:
        return (0.60 * feature_scores) + (0.25 * genre_mask) + (0.15 * year_scores)
    else:
        return (0.70 * feature_scores) + (0.30 * genre_mask)


def get_playlist_context(playlist_indices: list, df, text_embeddings) -> dict:
    songs = df.iloc[playlist_indices]
    mean_features = songs[AUDIO_FEATURES].mean().to_dict()
    genres = songs["genre"].unique().tolist()
    mean_text_vec = text_embeddings[playlist_indices].mean(axis=0, keepdims=True)
    return {"features": mean_features, "genres": genres, "text_vec": mean_text_vec}


def search_songs(query: str, df, limit: int = 5):
    q = query.lower()
    mask = (
        df["track_name"].str.lower().str.contains(q, na=False) |
        df["artist_name"].str.lower().str.contains(q, na=False)
    )
    positions = np.where(mask.values)[0][:limit]
    results = df.iloc[positions][["artist_name", "track_name", "genre", "energy", "valence", "tempo"]].copy()
    return results


def recommend(
    playlist_name: str,
    df,
    text_embeddings,
    top_k: int = 10,
    alpha: float = 0.9,
    playlist_indices: list = None,
    scaler=None,
    kmeans=None,
) -> tuple:
    playlist_indices = playlist_indices or []
    n = len(playlist_indices)

    name_weight = 1.0 if n == 0 else max(0.2, 1.0 / (1.0 + n))

    interpretation = interpret_playlist(playlist_name)
    year_range = interpretation.get("year_range")

    relevant_clusters = None
    if scaler is not None and kmeans is not None:
        relevant_clusters = get_relevant_clusters(interpretation["features"], scaler, kmeans)

    name_audio_scores = score_songs_by_audio(
        df, interpretation["features"], interpretation["genres"], relevant_clusters, year_range
    )
    name_text_vec = create_playlist_embedding(playlist_name)
    name_text_sim = cosine_similarity(name_text_vec, text_embeddings)[0]

    if n > 0:
        ctx = get_playlist_context(playlist_indices, df, text_embeddings)

        playlist_year_range = None
        if "year" in df.columns:
            mean_year = int(df.iloc[playlist_indices]["year"].mean())
            playlist_year_range = (mean_year - 10, mean_year + 10)

        playlist_clusters = None
        if scaler is not None and kmeans is not None:
            playlist_clusters = get_relevant_clusters(ctx["features"], scaler, kmeans)

        playlist_audio_scores = score_songs_by_audio(
            df, ctx["features"], ctx["genres"], playlist_clusters, playlist_year_range
        )
        playlist_text_sim = cosine_similarity(ctx["text_vec"], text_embeddings)[0]

        audio_scores = name_weight * name_audio_scores + (1 - name_weight) * playlist_audio_scores
        text_sim = name_weight * name_text_sim + (1 - name_weight) * playlist_text_sim
    else:
        audio_scores = name_audio_scores
        text_sim = name_text_sim

    text_sim_norm = (text_sim - text_sim.min()) / (text_sim.max() - text_sim.min() + 1e-8)
    final_scores = (alpha * audio_scores) + ((1 - alpha) * text_sim_norm)

    if n > 0:
        final_scores[playlist_indices] = -1.0

    top_idx = final_scores.argsort()[-top_k:][::-1]
    top_idx = top_idx[final_scores[top_idx] >= 0]

    results = df.iloc[top_idx][[
        "artist_name", "track_name", "genre",
        "energy", "danceability", "valence", "tempo",
        "acousticness", "key", "mode", "year",
    ]].copy()
    results["score"] = final_scores[top_idx].round(3)

    interpretation["name_weight"] = name_weight

    return results, interpretation
