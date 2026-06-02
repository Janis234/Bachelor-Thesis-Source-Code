import numpy as np


TEST_PLAYLISTS = [
    {
        "name": "Heavy Metal",
        "ground_truth": lambda df: df["genre"].str.contains("metal", case=False, na=False),
    },
    {
        "name": "Workout",
        "ground_truth": lambda df: df["genre"].str.contains("edm|hardstyle|hardcore|drum-and-bass|hard-rock|dubstep", case=False, na=False, regex=True),
    },
    {
        "name": "Sad Songs",
        "ground_truth": lambda df: df["genre"].str.contains("sad|emo|blues", case=False, na=False, regex=True),
    },
    {
        "name": "Hip Hop",
        "ground_truth": lambda df: df["genre"].str.contains("hip-hop|rap|trap", case=False, na=False, regex=True),
    },
    {
        "name": "Acoustic Chill",
        "ground_truth": lambda df: df["genre"].str.contains("acoustic|folk|singer-songwriter|chill", case=False, na=False, regex=True),
    },
    {
        "name": "Club Night",
        "ground_truth": lambda df: df["genre"].str.contains("club|house|techno|edm|disco|electro|deep-house|minimal-techno", case=False, na=False, regex=True),
    },
    {
        "name": "Sleep Music",
        "ground_truth": lambda df: df["genre"].str.contains("sleep|ambient|new-age|piano", case=False, na=False, regex=True),
    },
    {
        "name": "Punk Rock",
        "ground_truth": lambda df: df["genre"].str.contains("punk", case=False, na=False),
    },
    {
        "name": "Jazz Blues",
        "ground_truth": lambda df: df["genre"].str.contains("jazz|blues", case=False, na=False, regex=True),
    },
    {
        "name": "Country Drive",
        "ground_truth": lambda df: df["genre"].str.contains("country|folk|bluegrass", case=False, na=False, regex=True),
    },
    {
        "name": "Romantic Evening",
        "ground_truth": lambda df: df["genre"].str.contains("romance|soul", case=False, na=False, regex=True),
    },
    {
        "name": "Pop Hits",
        "ground_truth": lambda df: df["genre"].str.contains("pop", case=False, na=False),
    },
    {
        "name": "Focus Study",
        "ground_truth": lambda df: df["genre"].str.contains("classical|piano|ambient|new-age", case=False, na=False, regex=True),
    },
    {
        "name": "Tropical Beach",
        "ground_truth": lambda df: df["genre"].str.contains("reggae|afrobeat|salsa|samba|dancehall", case=False, na=False, regex=True),
    },
    {
        "name": "Classical Music",
        "ground_truth": lambda df: df["genre"].str.contains("classical|piano|opera", case=False, na=False, regex=True),
    },
]


def get_ground_truth_indices(df, playlist) -> set:
    mask = playlist["ground_truth"](df)
    return set(np.where(mask.values)[0].tolist())


HELD_OUT_PLAYLISTS = [
    {
        "name": "Headbanger Anthems",
        "ground_truth": lambda df: df["genre"].str.contains("metal", case=False, na=False),
    },
    {
        "name": "Gym Beast Mode",
        "ground_truth": lambda df: df["genre"].str.contains("edm|hardstyle|hardcore|drum-and-bass|hard-rock|dubstep", case=False, na=False, regex=True),
    },
    {
        "name": "Heartbreak Hour",
        "ground_truth": lambda df: df["genre"].str.contains("sad|emo|blues", case=False, na=False, regex=True),
    },
    {
        "name": "Trap Bangers",
        "ground_truth": lambda df: df["genre"].str.contains("hip-hop|rap|trap", case=False, na=False, regex=True),
    },
    {
        "name": "Coffee Shop",
        "ground_truth": lambda df: df["genre"].str.contains("acoustic|folk|singer-songwriter|chill", case=False, na=False, regex=True),
    },
    {
        "name": "Dance Floor",
        "ground_truth": lambda df: df["genre"].str.contains("club|house|techno|edm|disco|electro|deep-house|minimal-techno", case=False, na=False, regex=True),
    },
    {
        "name": "Bedtime",
        "ground_truth": lambda df: df["genre"].str.contains("sleep|ambient|new-age|piano", case=False, na=False, regex=True),
    },
    {
        "name": "Mosh Pit",
        "ground_truth": lambda df: df["genre"].str.contains("punk", case=False, na=False),
    },
    {
        "name": "Smoky Lounge",
        "ground_truth": lambda df: df["genre"].str.contains("jazz|blues", case=False, na=False, regex=True),
    },
    {
        "name": "Highway Cruise",
        "ground_truth": lambda df: df["genre"].str.contains("country|folk|bluegrass", case=False, na=False, regex=True),
    },
    {
        "name": "Date Night",
        "ground_truth": lambda df: df["genre"].str.contains("romance|soul", case=False, na=False, regex=True),
    },
    {
        "name": "Top 40 Bangers",
        "ground_truth": lambda df: df["genre"].str.contains("pop", case=False, na=False),
    },
    {
        "name": "Deep Work",
        "ground_truth": lambda df: df["genre"].str.contains("classical|piano|ambient|new-age", case=False, na=False, regex=True),
    },
    {
        "name": "Beach Vibes",
        "ground_truth": lambda df: df["genre"].str.contains("reggae|afrobeat|salsa|samba|dancehall", case=False, na=False, regex=True),
    },
    {
        "name": "Orchestra Hall",
        "ground_truth": lambda df: df["genre"].str.contains("classical|piano|opera", case=False, na=False, regex=True),
    },
]
