"""Wafer map triage tool — classify signatures, diagnose root-cause direction.

Run:  streamlit run app.py
Requires data/processed/labeled.npz and models/wafer_cnn.pt (see README).
"""

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src import data
from src.diagnosis import ROOT_CAUSES
from src.models import WaferCNN, maps_to_tensor

WAFER_CMAP = ListedColormap(["#f0f0f0", "#4c9be8", "#e8554c"])

st.set_page_config(page_title="Wafer Map Triage", page_icon="🎯", layout="wide")


@st.cache_resource
def load_model():
    model = WaferCNN(n_classes=len(data.CLASSES))
    model.load_state_dict(torch.load("models/wafer_cnn.pt", map_location="cpu", weights_only=True))
    return model.eval()


@st.cache_data
def load_maps():
    """Full labeled set locally; falls back to the bundled demo subset on servers."""
    path = data.PROCESSED if data.PROCESSED.exists() else data.DATA_DIR / "demo_sample.npz"
    npz = np.load(path)
    return npz["X"], npz["y"]


def predict(model, maps: np.ndarray) -> np.ndarray:
    """(N, 64, 64) -> (N, 9) class probabilities."""
    with torch.no_grad():
        return torch.softmax(model(maps_to_tensor(maps)), dim=1).numpy()


def draw_map(wafer_map, ax, title=None, title_color=None):
    ax.imshow(wafer_map, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9, color=title_color or "black")
    ax.axis("off")


model = load_model()
X, y = load_maps()
rng = np.random.default_rng()

st.title("Wafer Map Triage")
st.caption(
    "Model reads the fail-die pattern → names the signature → points at a root-cause direction. "
    "Demo runs on the public WM-811K dataset (811k production wafer maps); true labels shown for verification."
)

threshold = st.sidebar.slider(
    "Review threshold", 0.5, 0.99, 0.90, 0.01,
    help="Below this confidence the wafer is routed to engineer review instead of auto-tagged.",
)
tab_single, tab_lot, tab_cluster = st.tabs(["Single wafer", "Lot triage", "Cluster view"])


# --- Single wafer -----------------------------------------------------------
with tab_single:
    left, right = st.columns([1, 1.4])
    with left:
        if st.button("Sample a wafer", type="primary") or "single_idx" not in st.session_state:
            st.session_state.single_idx = int(rng.integers(0, len(X)))
        idx = st.session_state.single_idx
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        draw_map(X[idx], ax)
        st.pyplot(fig, width="content")
        st.caption(f"Wafer #{idx:,} — true label: **{data.CLASSES[y[idx]]}**")

    with right:
        probs = predict(model, X[idx][None])[0]
        top = int(probs.argmax())
        cls, conf = data.CLASSES[top], float(probs[top])

        if conf >= threshold:
            st.success(f"**{cls}** — confidence {conf:.1%} → auto-tag for SPC trending")
        else:
            st.warning(f"**{cls}?** — confidence {conf:.1%} below threshold → route to engineer review")

        info = ROOT_CAUSES[cls]
        st.markdown(f"**Root-cause direction:** {info['direction']}")
        st.markdown(f"**First checks:** {info['first_checks']}")

        order = np.argsort(probs)[::-1][:4]
        st.markdown("**Class probabilities**")
        for i in order:
            st.progress(float(probs[i]), text=f"{data.CLASSES[i]} — {probs[i]:.1%}")


# --- Lot triage -------------------------------------------------------------
with tab_lot:
    st.markdown(
        "Simulates a lot landing at sort: 25 wafers classified in one pass. "
        "Green = auto-tagged clean, red = signature detected, orange = needs human review."
    )
    if st.button("Triage a new lot", type="primary") or "lot_idx" not in st.session_state:
        st.session_state.lot_idx = rng.integers(0, len(X), size=25)
    lot = st.session_state.lot_idx
    probs = predict(model, X[lot])
    preds, confs = probs.argmax(1), probs.max(1)

    grid_col, sum_col = st.columns([1.6, 1])
    with grid_col:
        fig, axes = plt.subplots(5, 5, figsize=(7.5, 8))
        none_idx = data.CLASSES.index("none")
        for ax, i, p, c in zip(axes.ravel(), lot, preds, confs):
            if c < threshold:
                color, label = "#b47607", f"review ({data.CLASSES[p]}?)"
            elif p == none_idx:
                color, label = "#2b8a3e", "clean"
            else:
                color, label = "#c92a2a", data.CLASSES[p]
            draw_map(X[i], ax, title=label, title_color=color)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    with sum_col:
        st.markdown("**Lot summary**")
        auto = confs >= threshold
        n_clean = int(((preds == none_idx) & auto).sum())
        n_review = int((~auto).sum())
        n_flag = 25 - n_clean - n_review
        st.metric("Auto-cleared clean", n_clean)
        st.metric("Signatures flagged", n_flag)
        st.metric("Sent to review", n_review)
        if n_flag:
            st.markdown("**Flagged signatures**")
            for i, p, c in zip(lot, preds, confs):
                if c >= threshold and p != none_idx:
                    st.markdown(
                        f"- Wafer #{i:,}: **{data.CLASSES[p]}** ({c:.0%}) → {ROOT_CAUSES[data.CLASSES[p]]['direction']}"
                    )


# --- Cluster view -----------------------------------------------------------
with tab_cluster:
    st.markdown(
        "The CNN's learned embedding, projected to 2-D (PCA). Wafers with similar failure patterns land together — "
        "useful for spotting **new** signatures that don't fit the known classes: they show up as their own cluster."
    )
    n_sample = st.select_slider("Wafers to embed", [500, 1000, 2000, 4000], value=2000)
    if st.button("Compute clusters", type="primary") or "cluster_idx" not in st.session_state:
        # oversample patterned wafers so the view isn't 85% "none"
        pattern = np.flatnonzero(y != data.CLASSES.index("none"))
        clean = np.flatnonzero(y == data.CLASSES.index("none"))
        take = rng.permutation(np.concatenate([
            rng.choice(pattern, size=min(n_sample // 2, len(pattern)), replace=False),
            rng.choice(clean, size=n_sample // 2, replace=False),
        ]))
        st.session_state.cluster_idx = take
    take = st.session_state.cluster_idx

    with st.spinner("Embedding wafers..."):
        with torch.no_grad():
            emb = []
            for start in range(0, len(take), 512):
                batch = maps_to_tensor(X[take[start:start + 512]])
                emb.append(model.features(batch).mean(dim=(2, 3)))
            emb = torch.cat(emb).numpy()
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(emb)

    preds = predict(model, X[take]).argmax(1)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = plt.get_cmap("tab10")
    for c, cls in enumerate(data.CLASSES):
        mask = preds == c
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.6, color=palette(c % 10), label=cls)
    ax.legend(markerscale=2, fontsize=8, loc="best", ncols=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("CNN embedding of wafer maps (PCA, colored by predicted signature)", fontsize=10)
    st.pyplot(fig, width="stretch")
