import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key",
    "mode",
    "time_signature",
]

N_CLUSTERS = 20


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.drop_duplicates(subset="track_id")
    df = df.reset_index(drop=True)

    df["text"] = (
        df["track_name"].astype(str) + " " +
        df["artist_name"].astype(str) + " " +
        df["genre"].astype(str)
    )
    df["text"] = df["text"].str.lower()

    artist_counts = df.groupby("artist_name")["track_id"].transform("count")
    df["artist_freq"] = artist_counts / len(df)

    return df


def get_audio_features(df: pd.DataFrame):
    scaler = StandardScaler()
    audio_matrix = scaler.fit_transform(df[AUDIO_FEATURES])
    return audio_matrix, scaler


def fit_clusters(audio_matrix: np.ndarray, n_clusters: int = N_CLUSTERS):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(audio_matrix)
    return kmeans, labels


def preprocess_pipeline(path: str):
    df = load_data(path)
    audio_features, scaler = get_audio_features(df)
    kmeans, cluster_labels = fit_clusters(audio_features)
    df["cluster"] = cluster_labels
    return df, audio_features, scaler, kmeans
