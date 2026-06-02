import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_song_text_embeddings(text_list):
    embeddings = model.encode(text_list, show_progress_bar=True)
    return np.array(embeddings)


def create_playlist_embedding(playlist_name: str):
    return model.encode([playlist_name])


def combine_embeddings(audio_embeddings, text_embeddings):
    return np.hstack([audio_embeddings, text_embeddings])


def build_song_embeddings(df, audio_features):
    text_embeddings = create_song_text_embeddings(df["text"].tolist())
    combined_embeddings = combine_embeddings(audio_features, text_embeddings)
    return combined_embeddings, text_embeddings
