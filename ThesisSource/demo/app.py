import os
import sys
import urllib.parse
import numpy as np
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_pipeline
from src.embeddings import build_song_embeddings
from src.recommender import recommend, search_songs

st.set_page_config(page_title="AI Playlist", page_icon="🎵", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');

    /* ── Palette ─────────────────────────────────────────────────────────
       bg        #0c0c0c   page background
       surface   #161616   card surface
       border    #232323   subtle border
       text      #ededed   primary text
       muted     #8a8a8a   secondary text
       dim       #555      tertiary text
       accent    #1db954   single accent (Spotify green)
       ───────────────────────────────────────────────────────────────────── */

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #0c0c0c; }

    h1, h2, h3 { color: #ededed; letter-spacing: -0.01em; }
    h2 { font-weight: 700; font-size: 1.6rem; margin-bottom: 0.5rem; }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextInput input {
        background-color: #161616 !important;
        color: #f5f5f5 !important;
        border: 1px solid #232323 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 0.85rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextInput input:focus { border-color: #1db95488 !important; }
    .stTextInput input::placeholder { color: #888 !important; opacity: 1; }

    /* Buttons — covers st.button AND st.form_submit_button */
    .stButton > button,
    .stFormSubmitButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background: #1db954 !important;
        color: #000 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.1rem !important;
        font-size: 0.9rem !important;
        transition: background 0.15s ease;
    }
    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #1ed760 !important;
    }

    /* Card surface — shared by playlist card, in-list song, recommendation */
    .card {
        background: #161616;
        border: 1px solid #1f1f1f;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    .card:hover { border-color: #2c2c2c; }
    .card.is-hero {
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .card.is-hero:hover { border-color: #1db95466; }
    .card.is-faded { opacity: 0.45; }
    .card.is-rec:hover { border-color: #1db95466; }

    /* Typography ─ titles & meta */
    .t-title  { font-size: 0.95rem; font-weight: 600; color: #f5f5f5; line-height: 1.35; }
    .t-artist { font-size: 0.85rem; color: #c4c4c4; line-height: 1.3; }
    .t-meta   { font-size: 0.78rem; color: #a0a0a0; font-family: 'DM Mono', monospace; margin-top: 4px; }
    .t-hero-name { font-size: 1.1rem; font-weight: 700; color: #f5f5f5; }

    /* Tag chips — single unified style */
    .tag {
        display: inline-block;
        background: #232323;
        color: #d4d4d4;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.72rem;
        font-family: 'DM Mono', monospace;
        margin-right: 5px;
        margin-top: 4px;
    }
    .tag.is-accent { background: #1db95422; color: #1db954; }

    /* Stats row — feature numbers under each recommendation */
    .stat {
        color: #a8a8a8;
        font-size: 0.74rem;
        font-family: 'DM Mono', monospace;
        margin-top: 6px;
        letter-spacing: 0.01em;
    }
    .stat .sep { color: #4a4a4a; margin: 0 6px; }

    /* Vibe pill — shows matched anchor */
    .vibe-pill {
        display: inline-block;
        background: #1db95412;
        border: 1px solid #1db95433;
        color: #1db954;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.76rem;
        font-family: 'DM Mono', monospace;
        margin-bottom: 14px;
    }

    /* Play link — neutral, not screaming red */
    a.play-link {
        font-size: 0.75rem;
        color: #9a9a9a;
        text-decoration: none;
        margin-left: 6px;
    }
    a.play-link:hover { color: #1db954; }

    /* Section headings inside columns */
    .section-header {
        color: #f5f5f5;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 12px 0;
    }
    .section-header .muted { font-weight: 400; color: #a8a8a8; text-transform: none; letter-spacing: 0; }

    /* Influence indicator */
    .influence-row {
        font-size: 0.78rem;
        color: #c4c4c4;
        font-family: 'DM Mono', monospace;
        margin-bottom: 12px;
    }
    .influence-row .name-w  { color: #1db954; font-weight: 500; }
    .influence-row .pl-w    { color: #c4c4c4; }

    .empty-state {
        color: #a0a0a0;
        font-size: 0.9rem;
        padding: 28px 0;
        font-style: italic;
    }
    .divider { border-top: 1px solid #232323; margin: 18px 0; }

    /* ── Streamlit widget overrides (expander + sliders) ───────────────── */

    /* Expander header */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details summary p,
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p {
        color: #f5f5f5 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        background: transparent !important;
    }
    [data-testid="stExpander"] {
        background: #161616 !important;
        border: 1px solid #232323 !important;
        border-radius: 10px !important;
    }
    /* Kill Streamlit's default white-ish hover/focus tint on the summary */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active {
        background-color: transparent !important;
        outline: none !important;
    }
    [data-testid="stExpander"] summary:hover { background-color: #1c1c1c !important; }
    [data-testid="stExpander"] details > div {
        padding: 8px 4px !important;
    }

    /* Slider labels (e.g. "Number of results", "Audio vs Text weight") */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    .stSlider label,
    .stSlider label p {
        color: #e8e8e8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    /* Slider numeric value bubbles (above thumb & track endpoints) */
    .stSlider [data-baseweb="slider"] div[role="slider"] + div,
    .stSlider [data-baseweb="slider"] > div > div > div {
        color: #d4d4d4 !important;
        font-family: 'DM Mono', monospace !important;
    }
    /* Min/max labels at track ends */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #a0a0a0 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
    }
    /* Current value above thumb */
    .stSlider [data-testid="stThumbValue"] {
        color: #f5f5f5 !important;
        font-family: 'DM Mono', monospace !important;
        font-weight: 600 !important;
    }
    /* Slider track + thumb in accent green */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #1db954 !important;
        border-color: #1db954 !important;
    }

    /* Help tooltip icon next to slider labels */
    [data-testid="stTooltipIcon"] svg { color: #8a8a8a !important; }
    [data-testid="stTooltipIcon"]:hover svg { color: #1db954 !important; }
</style>
""", unsafe_allow_html=True)

EMBEDDINGS_PATH = "data/song_embeddings.npy"
TEXT_EMBEDDINGS_PATH = "data/text_embeddings.npy"


@st.cache_resource(show_spinner="Loading dataset… (first run only)")
def load_everything():
    df, audio_features, scaler, kmeans = preprocess_pipeline("data/spotify_data.csv")
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(TEXT_EMBEDDINGS_PATH):
        song_embeddings = np.load(EMBEDDINGS_PATH)
        text_embeddings = np.load(TEXT_EMBEDDINGS_PATH)
    else:
        song_embeddings, text_embeddings = build_song_embeddings(df, audio_features)
        np.save(EMBEDDINGS_PATH, song_embeddings)
        np.save(TEXT_EMBEDDINGS_PATH, text_embeddings)
    return df, song_embeddings, text_embeddings, scaler, kmeans


df, song_embeddings, text_embeddings, scaler, kmeans = load_everything()

if "playlists" not in st.session_state:
    st.session_state.playlists = []
if "view" not in st.session_state:
    st.session_state.view = "home"
if "current_pid" not in st.session_state:
    st.session_state.current_pid = None
if "recs" not in st.session_state:
    st.session_state.recs = {}
if "needs_refresh" not in st.session_state:
    st.session_state.needs_refresh = {}
if "creating" not in st.session_state:
    st.session_state.creating = False
if "settings" not in st.session_state:
    st.session_state.settings = {}


def get_playlist(pid):
    return next((p for p in st.session_state.playlists if p["id"] == pid), None)


def create_playlist(name: str) -> int:
    pid = len(st.session_state.playlists)
    st.session_state.playlists.append({
        "id": pid,
        "name": name,
        "songs": [],
        "indices": [],
    })
    return pid


def add_song(pid: int, idx: int, track_name: str, artist_name: str, genre: str):
    pl = get_playlist(pid)
    if idx not in pl["indices"]:
        pl["songs"].append({"idx": idx, "track_name": track_name, "artist_name": artist_name, "genre": genre})
        pl["indices"].append(idx)
        st.session_state.needs_refresh[pid] = True


def remove_song(pid: int, idx: int):
    pl = get_playlist(pid)
    pl["songs"] = [s for s in pl["songs"] if s["idx"] != idx]
    pl["indices"] = [i for i in pl["indices"] if i != idx]
    st.session_state.needs_refresh[pid] = True


def delete_playlist(pid: int):
    st.session_state.playlists = [p for p in st.session_state.playlists if p["id"] != pid]
    st.session_state.recs.pop(pid, None)
    st.session_state.needs_refresh.pop(pid, None)
    st.session_state.settings.pop(pid, None)


def refresh_recs(pid: int):
    pl = get_playlist(pid)
    s = st.session_state.settings.get(pid, {"top_k": 10, "alpha": 0.9})
    results, interpretation = recommend(
        pl["name"], df, text_embeddings,
        top_k=s["top_k"], alpha=s["alpha"],
        playlist_indices=pl["indices"],
        scaler=scaler,
        kmeans=kmeans,
    )
    st.session_state.recs[pid] = {"results": results, "interpretation": interpretation}
    st.session_state.needs_refresh[pid] = False


_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def fmt_key(key, mode) -> str:
    name = _KEY_NAMES[int(key) % 12]
    tonality = "maj" if int(mode) == 1 else "min"
    return f"{name} {tonality}"

def yt_link(track_name: str, artist_name: str) -> str:
    q = urllib.parse.quote(f"{track_name} {artist_name}")
    return f"https://www.youtube.com/results?search_query={q}"


def render_home():
    st.markdown("## Your Playlists")

    playlists = st.session_state.playlists

    if playlists:
        cols = st.columns(3)
        for pl in playlists:
            n = len(pl["songs"])
            with cols[pl["id"] % 3]:
                st.markdown(
                    f'<div class="card is-hero">'
                    f'<div class="t-hero-name">{pl["name"]}</div>'
                    f'<div class="t-meta">{n} song{"s" if n != 1 else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                btn_open, btn_del = st.columns([3, 1])
                with btn_open:
                    if st.button("Open", key=f"open_{pl['id']}", use_container_width=True):
                        if pl["id"] not in st.session_state.recs:
                            with st.spinner("Loading recommendations…"):
                                refresh_recs(pl["id"])
                        st.session_state.current_pid = pl["id"]
                        st.session_state.view = "playlist"
                        st.rerun()
                with btn_del:
                    if st.button("🗑", key=f"del_{pl['id']}", use_container_width=True, help="Delete playlist"):
                        delete_playlist(pl["id"])
                        st.rerun()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    else:
        st.markdown('<p class="empty-state">No playlists yet. Create your first one below.</p>', unsafe_allow_html=True)

    if not st.session_state.creating:
        if st.button("＋ New Playlist"):
            st.session_state.creating = True
            st.rerun()
    else:
        with st.form("create_form", clear_on_submit=True):
            name = st.text_input("Playlist name", placeholder='"Sunday Morning Coffee", "Gym Session", "3am Drive"')
            col_ok, col_cancel = st.columns([2, 1])
            with col_ok:
                submitted = st.form_submit_button("Create", use_container_width=True)
            with col_cancel:
                cancelled = st.form_submit_button("Cancel", use_container_width=True)

        if submitted and name.strip():
            pid = create_playlist(name.strip())
            st.session_state.creating = False
            with st.spinner("Loading initial recommendations…"):
                refresh_recs(pid)
            st.session_state.current_pid = pid
            st.session_state.view = "playlist"
            st.rerun()

        if cancelled:
            st.session_state.creating = False
            st.rerun()


def render_playlist(pid: int):
    pl = get_playlist(pid)
    if not pl:
        st.session_state.view = "home"
        st.rerun()
        return

    h_col, title_col = st.columns([1, 9])
    with h_col:
        if st.button("← Back"):
            st.session_state.view = "home"
            st.session_state.current_pid = None
            st.rerun()
    with title_col:
        st.markdown(f"## {pl['name']}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([5, 4], gap="large")

    with left_col:
        n = len(pl["songs"])
        st.markdown(
            f'<p class="section-header">Playlist <span class="muted">· {n} song{"s" if n != 1 else ""}</span></p>',
            unsafe_allow_html=True
        )

        if not pl["songs"]:
            st.markdown('<p class="empty-state">No songs yet — add from recommendations →</p>', unsafe_allow_html=True)
        else:
            for song in pl["songs"]:
                s_col, btn_col = st.columns([9, 1])
                with s_col:
                    st.markdown(
                        f'<div class="card">'
                        f'<span class="t-title">{song["track_name"]}</span>'
                        f'<a class="play-link" href="{yt_link(song["track_name"], song["artist_name"])}" target="_blank">▶ play</a><br>'
                        f'<span class="t-artist">{song["artist_name"]}</span>'
                        f'<span class="tag" style="margin-left:8px">{song["genre"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with btn_col:
                    st.markdown('<div style="padding-top:10px">', unsafe_allow_html=True)
                    if st.button("×", key=f"rm_{pid}_{song['idx']}", help="Remove"):
                        remove_song(pid, song["idx"])
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        s = st.session_state.settings.get(pid, {"top_k": 10, "alpha": 0.9})
        with st.expander("⚙ Settings"):
            new_top_k = st.slider("Number of results", 5, 20, s["top_k"], key=f"sl_topk_{pid}")
            new_alpha = st.slider(
                "Audio vs Text weight", 0.0, 1.0, s["alpha"], 0.05,
                help="Higher = audio features. Lower = name/text similarity.",
                key=f"sl_alpha_{pid}"
            )
            if st.button("Save", key=f"save_settings_{pid}"):
                st.session_state.settings[pid] = {"top_k": new_top_k, "alpha": new_alpha}
                with st.spinner("Applying settings…"):
                    refresh_recs(pid)
                st.rerun()

        rec_data = st.session_state.recs.get(pid)
        needs_refresh = st.session_state.needs_refresh.get(pid, False)

        r_head, r_btn = st.columns([5, 3])
        with r_head:
            st.markdown('<p class="section-header">Recommendations</p>', unsafe_allow_html=True)
        with r_btn:
            btn_label = "🔄 Update" if needs_refresh else "🔄 Refresh"
            btn_style = "border: 1px solid #1db954 !important;" if needs_refresh else ""
            if st.button(btn_label, key=f"refresh_{pid}", use_container_width=True):
                with st.spinner("Updating…"):
                    refresh_recs(pid)
                st.rerun()

        if needs_refresh:
            st.markdown('<p style="font-size:0.78rem;color:#1db954;margin:-8px 0 8px 0">Songs added — click Update ↑</p>', unsafe_allow_html=True)

        if rec_data:
            results = rec_data["results"]
            interpretation = rec_data["interpretation"]
            name_weight = interpretation["name_weight"]

            st.markdown(f'<div class="vibe-pill">{interpretation["matched_anchors"][0]}</div>', unsafe_allow_html=True)
            if pl["songs"]:
                st.markdown(
                    f'<div class="influence-row">'
                    f'<span class="name-w">name {name_weight:.0%}</span> · '
                    f'<span class="pl-w">playlist {1 - name_weight:.0%}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            for i, row in enumerate(results.itertuples(), 1):
                idx = int(row.Index)
                already_added = idx in pl["indices"]
                rc_col, rb_col = st.columns([8, 1])
                with rc_col:
                    fade_cls = " is-faded" if already_added else ""
                    st.markdown(
                        f'<div class="card is-rec{fade_cls}">'
                        f'<span class="t-title">{row.track_name}</span>'
                        f'<a class="play-link" href="{yt_link(row.track_name, row.artist_name)}" target="_blank">▶ play</a><br>'
                        f'<span class="t-artist">{row.artist_name}</span><br>'
                        f'<span class="tag is-accent">{row.genre}</span>'
                        f'<span class="tag">{int(row.year)}</span>'
                        f'<span class="tag">{fmt_key(row.key, row.mode)}</span>'
                        f'<div class="stat">'
                        f'nrg {row.energy:.2f}<span class="sep">·</span>'
                        f'val {row.valence:.2f}<span class="sep">·</span>'
                        f'dance {row.danceability:.2f}<span class="sep">·</span>'
                        f'acou {row.acousticness:.2f}<span class="sep">·</span>'
                        f'{row.tempo:.0f} bpm<span class="sep">·</span>'
                        f'score {row.score:.3f}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with rb_col:
                    st.markdown('<div style="padding-top:10px">', unsafe_allow_html=True)
                    if already_added:
                        st.markdown('<span style="color:#1db954;font-size:1rem">✓</span>', unsafe_allow_html=True)
                    else:
                        if st.button("＋", key=f"add_{pid}_{idx}_{i}", help="Add to playlist"):
                            add_song(pid, idx, row.track_name, row.artist_name, row.genre)
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Search &amp; add</p>', unsafe_allow_html=True)
        search_q = st.text_input(
            "Song or artist",
            key=f"search_{pid}",
            placeholder="input song...",
            label_visibility="collapsed"
        )

        if search_q.strip():
            hits = search_songs(search_q.strip(), df, limit=6)
            if hits.empty:
                st.markdown('<p class="empty-state">No results found.</p>', unsafe_allow_html=True)
            else:
                for row in hits.itertuples():
                    idx = int(row.Index)
                    already_added = idx in pl["indices"]
                    h_col, b_col = st.columns([8, 1])
                    with h_col:
                        check = '<span style="color:#1db954">✓</span> ' if already_added else ''
                        st.markdown(
                            f'<div class="card{" is-faded" if already_added else ""}">'
                            f'{check}<span class="t-title">{row.track_name}</span>'
                            f'<a class="play-link" href="{yt_link(row.track_name, row.artist_name)}" target="_blank">▶ play</a><br>'
                            f'<span class="t-artist">{row.artist_name}</span>'
                            f'<span class="tag" style="margin-left:8px">{row.genre}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    with b_col:
                        if not already_added:
                            if st.button("＋", key=f"sadd_{pid}_{idx}", help="Add"):
                                add_song(pid, idx, row.track_name, row.artist_name, row.genre)
                                st.rerun()


if st.session_state.view == "home":
    render_home()
elif st.session_state.view == "playlist":
    render_playlist(st.session_state.current_pid)
