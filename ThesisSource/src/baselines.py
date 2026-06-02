import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender import (
    interpret_playlist,
    score_songs_by_audio,
    get_relevant_clusters,
    AUDIO_FEATURES,
)
from src.embeddings import create_playlist_embedding
from src.custom_encoder import encode_target_features


_RESULT_COLUMNS = [
    "artist_name", "track_name", "genre",
    "energy", "danceability", "valence", "tempo",
    "acousticness", "key", "mode", "year",
]


def _build_results(df, indices, scores):
    indices = list(indices)
    results = df.iloc[indices][_RESULT_COLUMNS].copy()
    results["score"] = np.round(np.asarray(scores, dtype=float), 3)
    return results


def _take_top_k_excluding(sorted_idx, exclude_set, top_k):
    out = []
    for i in sorted_idx:
        i = int(i)
        if i not in exclude_set:
            out.append(i)
            if len(out) == top_k:
                break
    return np.array(out, dtype=int)


def recommend_random(
    playlist_name: str, df, text_embeddings=None,
    top_k: int = 10, playlist_indices: list = None, seed: int = 42, **kwargs,
) -> tuple:
    rng = np.random.default_rng(seed)
    excluded = set(playlist_indices or [])
    pool = np.array([i for i in range(len(df)) if i not in excluded])
    chosen = rng.choice(pool, size=top_k, replace=False)
    return _build_results(df, chosen, np.zeros(top_k)), {"baseline": "random"}


def recommend_popularity(
    playlist_name: str, df, text_embeddings=None,
    top_k: int = 10, playlist_indices: list = None, **kwargs,
) -> tuple:
    excluded = set(playlist_indices or [])
    sorted_idx = df["popularity"].values.argsort()[::-1]
    chosen = _take_top_k_excluding(sorted_idx, excluded, top_k)
    scores = df["popularity"].values[chosen] / 100.0
    return _build_results(df, chosen, scores), {"baseline": "popularity"}


def recommend_genre_only(
    playlist_name: str, df, text_embeddings=None,
    top_k: int = 10, playlist_indices: list = None, seed: int = 42, **kwargs,
) -> tuple:
    rng = np.random.default_rng(seed)
    excluded = set(playlist_indices or [])
    interpretation = interpret_playlist(playlist_name)
    allowed_genres = interpretation["genres"]

    keywords = set()
    for g in allowed_genres:
        for part in g.replace("-", " ").split():
            if len(part) >= 3:
                keywords.add(part)

    if keywords:
        pattern = "|".join(keywords)
        mask = df["genre"].str.contains(pattern, case=False, na=False).values
    else:
        mask = df["genre"].isin(allowed_genres).values

    pool = np.array([i for i in np.where(mask)[0] if i not in excluded])
    if len(pool) == 0:
        return recommend_random(
            playlist_name, df, text_embeddings, top_k=top_k,
            playlist_indices=playlist_indices, seed=seed,
        )

    size = min(top_k, len(pool))
    chosen = rng.choice(pool, size=size, replace=False)
    return _build_results(df, chosen, np.zeros(size)), {
        "baseline": "genre_only",
        "genres": allowed_genres,
    }


def recommend_audio_only(
    playlist_name: str, df, text_embeddings=None,
    top_k: int = 10, playlist_indices: list = None,
    scaler=None, kmeans=None, **kwargs,
) -> tuple:
    interpretation = interpret_playlist(playlist_name)
    year_range = interpretation.get("year_range")

    relevant_clusters = None
    if scaler is not None and kmeans is not None:
        relevant_clusters = get_relevant_clusters(interpretation["features"], scaler, kmeans)

    scores = score_songs_by_audio(
        df, interpretation["features"], interpretation["genres"],
        relevant_clusters, year_range,
    )

    if playlist_indices:
        scores = scores.copy()
        scores[list(playlist_indices)] = -1.0

    top_idx = scores.argsort()[-top_k:][::-1]
    top_idx = top_idx[scores[top_idx] >= 0]

    return _build_results(df, top_idx, scores[top_idx]), {
        "baseline": "audio_only",
        "matched_anchors": interpretation["matched_anchors"],
    }


def recommend_text_only(
    playlist_name: str, df, text_embeddings,
    top_k: int = 10, playlist_indices: list = None, **kwargs,
) -> tuple:
    name_vec = create_playlist_embedding(playlist_name)
    text_sim = cosine_similarity(name_vec, text_embeddings)[0]

    if playlist_indices:
        text_sim = text_sim.copy()
        text_sim[list(playlist_indices)] = -1.0

    top_idx = text_sim.argsort()[-top_k:][::-1]
    top_idx = top_idx[text_sim[top_idx] >= 0]

    return _build_results(df, top_idx, text_sim[top_idx]), {"baseline": "text_only"}


def recommend_custom_nn(
    playlist_name: str, df, text_embeddings=None,
    top_k: int = 10, playlist_indices: list = None,
    scaler=None, custom_encoder=None, custom_embeddings=None, **kwargs,
) -> tuple:
    if custom_encoder is None or custom_embeddings is None or scaler is None:
        raise ValueError(
            "recommend_custom_nn requires custom_encoder, custom_embeddings, and scaler"
        )

    interpretation = interpret_playlist(playlist_name)
    target_vec = np.array([[interpretation["features"][f] for f in AUDIO_FEATURES]])
    target_vec_std = scaler.transform(target_vec)
    target_emb = encode_target_features(custom_encoder, target_vec_std)

    sim = cosine_similarity(target_emb, custom_embeddings)[0]

    if playlist_indices:
        sim = sim.copy()
        sim[list(playlist_indices)] = -1.0

    top_idx = sim.argsort()[-top_k:][::-1]
    top_idx = top_idx[sim[top_idx] >= 0]

    return _build_results(df, top_idx, sim[top_idx]), {
        "baseline": "custom_nn",
        "matched_anchors": interpretation["matched_anchors"],
    }


def recommend_hstack_cosine(
    playlist_name: str, df, text_embeddings=None,
    top_k: int = 10, playlist_indices: list = None,
    scaler=None, song_embeddings=None, **kwargs,
) -> tuple:
    if scaler is None or song_embeddings is None:
        raise ValueError(
            "recommend_hstack_cosine requires scaler and song_embeddings kwargs"
        )

    interpretation = interpret_playlist(playlist_name)
    target_audio = np.array([[interpretation["features"][f] for f in AUDIO_FEATURES]])
    target_audio_std = scaler.transform(target_audio)

    target_text = create_playlist_embedding(playlist_name)

    query = np.hstack([target_audio_std, target_text])

    sim = cosine_similarity(query, song_embeddings)[0]

    if playlist_indices:
        sim = sim.copy()
        sim[list(playlist_indices)] = -1.0

    top_idx = sim.argsort()[-top_k:][::-1]
    top_idx = top_idx[sim[top_idx] >= 0]

    return _build_results(df, top_idx, sim[top_idx]), {
        "baseline": "hstack_cosine",
        "matched_anchors": interpretation["matched_anchors"],
    }


BASELINES = {
    "random":         recommend_random,
    "popularity":     recommend_popularity,
    "genre_only":     recommend_genre_only,
    "audio_only":     recommend_audio_only,
    "text_only":      recommend_text_only,
    "custom_nn":      recommend_custom_nn,
    "hstack_cosine":  recommend_hstack_cosine,
}
