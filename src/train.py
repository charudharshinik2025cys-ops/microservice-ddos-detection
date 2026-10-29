"""Train GCN, Random Forest, and threshold baseline; split by whole scenarios."""
import csv
import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .common import CLASSES, DATA_DIR, FEATURES, OUTPUT_DIR, ensure_dirs, load_config


def normalized_adjacency(n, edges, remove_edges=False):
    a = np.eye(n, dtype=float)
    if not remove_edges:
        for src, dst in edges:
            # A called service aggregates from its callers so effects can flow downstream.
            a[dst, src] = 1.0
    degree = a.sum(axis=1)
    return a / degree[:, None]


def fit_gcn(x, y, adjacency, cfg, seed):
    """Small two-layer GCN with explicit NumPy backpropagation for portability."""
    rng = np.random.default_rng(seed)
    n, f = x.shape[1:]
    h, c = cfg["hidden_units"], len(CLASSES)
    w0 = rng.normal(0, np.sqrt(2 / f), (f, h)); w1 = rng.normal(0, np.sqrt(2 / h), (h, c))
    b0, b1 = np.zeros(h), np.zeros(c)
    eye = np.eye(c)[y]
    for _ in range(cfg["epochs"]):
        ax = np.einsum("ij,bjf->bif", adjacency, x)
        pre = ax @ w0 + b0
        hidden = np.maximum(pre, 0)
        ah = np.einsum("ij,bjh->bih", adjacency, hidden)
        logits = ah @ w1 + b1
        logits -= logits.max(axis=2, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=2, keepdims=True)
        grad = (probs - eye) / (len(x) * n)
        gw1 = np.einsum("bih,bic->hc", ah, grad); gb1 = grad.sum(axis=(0, 1))
        gah = grad @ w1.T
        gh = np.einsum("ji,bjh->bih", adjacency, gah)
        gp = gh * (pre > 0)
        gw0 = np.einsum("bif,bih->fh", ax, gp); gb0 = gp.sum(axis=(0, 1))
        rate = cfg["learning_rate"]
        w0 -= rate * gw0; b0 -= rate * gb0; w1 -= rate * gw1; b1 -= rate * gb1
    return {"w0": w0, "b0": b0, "w1": w1, "b1": b1}


def main():
    cfg = load_config(); ensure_dirs()
    path = DATA_DIR / "synthetic_telemetry.csv"
    if not path.exists():
        raise SystemExit("Synthetic data is missing. Run: python -m src.generate_data")
    with path.open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
    services = cfg["services"]; service_ix = {s: i for i, s in enumerate(services)}
    scenarios = sorted({int(r["scenario_id"]) for r in rows})
    rng = np.random.default_rng(cfg["seed"]); rng.shuffle(scenarios)
    cut = max(1, int(.70 * len(scenarios)))
    train_ids, test_ids = set(scenarios[:cut]), set(scenarios[cut:])
    # Convert rows to one graph sample per complete time window.
    samples = {}
    for row in rows:
        key = (int(row["scenario_id"]), int(row["window"]))
        samples.setdefault(key, [np.zeros((len(services), len(FEATURES))), np.zeros(len(services), dtype=int), row])
        ix = service_ix[row["service"]]
        samples[key][0][ix] = [float(row[k]) for k in FEATURES]
        samples[key][1][ix] = CLASSES.index(row["label"])
    keys = sorted(samples)
    raw = np.stack([samples[k][0] for k in keys]); labels = np.stack([samples[k][1] for k in keys])
    train_mask = np.array([k[0] in train_ids for k in keys]); test_mask = ~train_mask
    scaler = StandardScaler().fit(raw[train_mask].reshape(-1, len(FEATURES)))
    x = scaler.transform(raw.reshape(-1, len(FEATURES))).reshape(raw.shape)
    flat_x, flat_y = x[train_mask].reshape(-1, len(FEATURES)), labels[train_mask].reshape(-1)
    rf = RandomForestClassifier(n_estimators=180, class_weight="balanced", random_state=cfg["seed"], n_jobs=-1)
    rf.fit(flat_x, flat_y)
    with (OUTPUT_DIR / "random_forest.pkl").open("wb") as f: pickle.dump(rf, f)
    with (OUTPUT_DIR / "scaler.pkl").open("wb") as f: pickle.dump(scaler, f)
    graph = json.loads((DATA_DIR / "service_graph.json").read_text(encoding="utf-8"))
    edge_ix = [(service_ix[a], service_ix[b]) for a, b in graph["edges"]]
    weights = fit_gcn(x[train_mask], labels[train_mask], normalized_adjacency(len(services), edge_ix), cfg, cfg["seed"])
    np.savez(OUTPUT_DIR / "gcn_weights.npz", **weights)
    np.savez(OUTPUT_DIR / "split.npz", test_keys=np.array(keys, dtype=object)[test_mask],
             test_x=x[test_mask], test_y=labels[test_mask], train_scenarios=np.array(sorted(train_ids)),
             test_scenarios=np.array(sorted(test_ids)), raw_test=raw[test_mask])
    (OUTPUT_DIR / "model_config.json").write_text(json.dumps({"services": services, "features": FEATURES,
        "classes": CLASSES, "seed": cfg["seed"], "epochs": cfg["epochs"]}, indent=2), encoding="utf-8")
    print(f"Training complete. Scenario split: {len(train_ids)} train / {len(test_ids)} test.")


if __name__ == "__main__": main()
