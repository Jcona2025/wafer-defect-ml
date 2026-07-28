"""Wafer Map Triage — Flask app.

Run:  python app.py            (dev, http://localhost:8601)
      gunicorn -w 1 -b 127.0.0.1:8601 app:app   (production)

Requires models/wafer_cnn.pt and either data/processed/labeled.npz (full)
or the bundled data/demo_sample.npz.
"""

import threading

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from src import data
from src.diagnosis import ROOT_CAUSES
from src.models import WaferCNN, maps_to_tensor

app = Flask(__name__)
torch.set_num_threads(2)


def _load_maps():
    path = data.PROCESSED if data.PROCESSED.exists() else data.DATA_DIR / "demo_sample.npz"
    npz = np.load(path)
    return npz["X"], npz["y"]


X, Y = _load_maps()
MODEL = WaferCNN(n_classes=len(data.CLASSES))
MODEL.load_state_dict(torch.load("models/wafer_cnn.pt", map_location="cpu", weights_only=True))
MODEL.eval()
RNG = np.random.default_rng()
NONE_IDX = data.CLASSES.index("none")


def predict(maps: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(MODEL(maps_to_tensor(maps)), dim=1).numpy()


def encode_map(m: np.ndarray) -> str:
    """(64, 64) of {0,1,2} -> 4096-char string of '0'/'1'/'2' for compact JSON."""
    return (m.ravel() + 48).astype(np.uint8).tobytes().decode("ascii")


def wafer_payload(idx: int, probs: np.ndarray) -> dict:
    order = np.argsort(probs)[::-1]
    return {
        "idx": int(idx),
        "map": encode_map(X[idx]),
        "true_label": data.CLASSES[Y[idx]],
        "pred": data.CLASSES[order[0]],
        "conf": round(float(probs[order[0]]), 4),
        "probs": [
            {"cls": data.CLASSES[i], "p": round(float(probs[i]), 4)} for i in order[:4]
        ],
    }


# --- cluster embedding: computed once in the background at startup ----------
CLUSTER: dict = {"ready": False}


def _compute_cluster(n_sample: int = 2500):
    pattern = np.flatnonzero(Y != NONE_IDX)
    clean = np.flatnonzero(Y == NONE_IDX)
    take = RNG.permutation(np.concatenate([
        RNG.choice(pattern, size=min(n_sample // 2, len(pattern)), replace=False),
        RNG.choice(clean, size=n_sample // 2, replace=False),
    ]))
    embs = []
    with torch.no_grad():
        for start in range(0, len(take), 256):
            batch = maps_to_tensor(X[take[start:start + 256]])
            embs.append(MODEL.features(batch).mean(dim=(2, 3)))
    emb = torch.cat(embs).numpy()

    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(emb)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))  # normalize to [0,1] for canvas
    preds = predict(X[take]).argmax(1)
    CLUSTER.update({
        "ready": True,
        "points": [
            [round(float(px), 4), round(float(py), 4), int(p), int(i)]
            for (px, py), p, i in zip(xy, preds, take)
        ],
    })


threading.Thread(target=_compute_cluster, daemon=True).start()


# --- routes -----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "classes": data.CLASSES,
        "root_causes": ROOT_CAUSES,
        "n_wafers": len(Y),
    })


@app.route("/api/wafer")
def wafer():
    cls = request.args.get("cls", "")
    if cls in data.CLASSES:
        idx = int(RNG.choice(np.flatnonzero(Y == data.CLASSES.index(cls))))
    elif cls == "any-pattern":
        idx = int(RNG.choice(np.flatnonzero(Y != NONE_IDX)))
    else:
        idx = int(RNG.integers(0, len(Y)))
    probs = predict(X[idx][None])[0]
    return jsonify(wafer_payload(idx, probs))


@app.route("/api/wafer/<int:idx>")
def wafer_by_idx(idx):
    if not 0 <= idx < len(Y):
        return jsonify({"error": "index out of range"}), 404
    probs = predict(X[idx][None])[0]
    return jsonify(wafer_payload(idx, probs))


@app.route("/api/lot")
def lot():
    idxs = RNG.integers(0, len(Y), size=25)
    probs = predict(X[idxs])
    return jsonify({"wafers": [wafer_payload(i, p) for i, p in zip(idxs, probs)]})


@app.route("/api/cluster")
def cluster():
    return jsonify(CLUSTER)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8601, debug=False)
