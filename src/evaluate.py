"""Evaluate saved models and create machine-readable metrics and plots."""
import csv
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .common import CLASSES, DATA_DIR, FEATURES, OUTPUT_DIR, ensure_dirs, load_config
from .train import normalized_adjacency


def predict_gcn(x, a, weights):
    hidden = np.maximum(np.einsum("ij,bjf->bif", a, x) @ weights["w0"] + weights["b0"], 0)
    logits = np.einsum("ij,bjh->bih", a, hidden) @ weights["w1"] + weights["b1"]
    return logits.argmax(axis=2)


def metrics(y, pred):
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    return {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": support.tolist(),
            "confusion_matrix": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()}


def write_dashboard(result):
    """Render the dashboard template with the configured services and call graph."""
    del result  # The live demo reads fresh predictions from the local endpoint.
    cfg = load_config()
    graph = json.loads((DATA_DIR / "service_graph.json").read_text(encoding="utf-8"))
    template = Path(__file__).with_name("dashboard_template.html").read_text(encoding="utf-8")
    options = "".join(f'<option value="{service}">{service.title()}</option>' for service in cfg["services"])
    page = (template.replace("__SERVICE_OPTIONS__", options)
            .replace("__SERVICE_DATA__", json.dumps(cfg["services"], separators=(",", ":")))
            .replace("__EDGE_DATA__", json.dumps(graph["edges"], separators=(",", ":"))))
    (OUTPUT_DIR / "dashboard.html").write_text(page, encoding="utf-8")


def main():
    ensure_dirs(); cfg = load_config()
    with np.load(OUTPUT_DIR / "split.npz", allow_pickle=True) as data:
        keys, x, y = data["test_keys"], data["test_x"], data["test_y"]
        raw, test_ids = data["raw_test"], set(map(int, data["test_scenarios"]))
    flat_x = x.reshape(-1, len(FEATURES)); flat_y = y.reshape(-1)
    with (OUTPUT_DIR / "random_forest.pkl").open("rb") as f: rf = pickle.load(f)
    rf_pred = rf.predict(flat_x).reshape(y.shape)
    weights = dict(np.load(OUTPUT_DIR / "gcn_weights.npz"))
    graph = json.loads((DATA_DIR / "service_graph.json").read_text(encoding="utf-8"))
    service_ix = {s: i for i, s in enumerate(cfg["services"])}
    edges = [(service_ix[a], service_ix[b]) for a, b in graph["edges"]]
    a = normalized_adjacency(len(cfg["services"]), edges)
    gcn_pred = predict_gcn(x, a, weights)
    ablated_pred = predict_gcn(x, normalized_adjacency(len(cfg["services"]), edges, True), weights)
    # Simple, transparent thresholds; only the directly anomalous service receives a positive label.
    threshold = np.zeros_like(y)
    direct = (raw[:, :, 0] > 250) | (raw[:, :, 1] > 190) | (raw[:, :, 2] > .18) | (raw[:, :, 3] > .08) | (raw[:, :, 4] > 78)
    threshold[direct] = CLASSES.index("directly_targeted")
    model_preds = {"gcn": gcn_pred, "random_forest": rf_pred, "threshold": threshold, "gcn_without_edges": ablated_pred}
    scenario_info = {int(o["scenario_id"]): o for o in json.loads(
        (DATA_DIR / "scenarios.json").read_text(encoding="utf-8"))}
    result = {"data_notice": "All data and results are synthetic simulation outputs; they do not establish real-world attack performance.",
              "split": "complete scenarios; preprocessing fitted on training scenarios only", "class_names": CLASSES, "models": {}}
    for name, pred in model_preds.items():
        flat_pred = pred.reshape(-1)
        class_metrics = metrics(flat_y, flat_pred)
        normal_rows, spike_rows = [], []
        for j, (sid, _) in enumerate(keys):
            kind = scenario_info[int(sid)]["scenario_type"]
            if kind == "normal": normal_rows.extend(pred[j].tolist())
            elif kind == "spike": spike_rows.extend(pred[j].tolist())
        fpr_normal = float(np.mean(np.array(normal_rows) != 0)) if normal_rows else None
        fpr_spike = float(np.mean(np.array(spike_rows) != 0)) if spike_rows else None
        local_ok, delays = [], []
        by_scenario = {}
        for j, (sid, win) in enumerate(keys): by_scenario.setdefault(int(sid), []).append((int(win), j))
        for sid, entries in by_scenario.items():
            info = scenario_info[sid]
            if info["scenario_type"] != "attack": continue
            attack_entries = [(w, j) for w, j in entries if info["event_start"] <= w < info["event_end"]]
            if not attack_entries: continue
            true = info["target_service"]
            target_votes = [cfg["services"][int(np.argmax(pred[j] == 1))] for _, j in attack_entries if np.any(pred[j] == 1)]
            local_ok.append(bool(target_votes and max(set(target_votes), key=target_votes.count) == true))
            first = next((w for w, j in attack_entries if pred[j][service_ix[true]] == 1), None)
            delays.append((first - int(info["event_start"])) if first is not None else None)
        result["models"][name] = {"per_class": {CLASSES[i]: {"precision": class_metrics["precision"][i], "recall": class_metrics["recall"][i], "f1": class_metrics["f1"][i], "support": class_metrics["support"][i]} for i in range(len(CLASSES))},
            "confusion_matrix": class_metrics["confusion_matrix"], "false_positive_rate_normal": fpr_normal,
            "false_positive_rate_legitimate_spike": fpr_spike,
            "target_service_localization_accuracy": float(np.mean(local_ok)) if local_ok else None,
            "mean_detection_delay_windows": float(np.mean([d for d in delays if d is not None])) if any(d is not None for d in delays) else None,
            "detection_delay_definition": "First attack window where the true target is predicted directly_targeted, minus attack start window; missed targets are omitted."}
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "evaluation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["model", "class", "precision", "recall", "f1", "support"])
        for name, info in result["models"].items():
            for cls, values in info["per_class"].items(): writer.writerow([name, cls, values["precision"], values["recall"], values["f1"], values["support"]])
    fig, axes = plt.subplots(1, len(model_preds), figsize=(16, 4))
    for ax, (name, pred) in zip(axes, model_preds.items()):
        cm = confusion_matrix(flat_y, pred.reshape(-1), labels=range(len(CLASSES)))
        ax.imshow(cm, cmap="Blues"); ax.set_title(name.replace("_", " ")); ax.set_xticks(range(3), ["normal", "target", "indirect"], rotation=35, ha="right"); ax.set_yticks(range(3), ["normal", "target", "indirect"])
        for (i, j), value in np.ndenumerate(cm): ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160); plt.close(fig)
    test_kinds = {kind: [sid for sid in sorted(test_ids) if scenario_info[sid]["scenario_type"] == kind]
                  for kind in ("normal", "spike", "attack")}
    selected = [test_kinds[kind][0] for kind in ("normal", "spike", "attack") if test_kinds[kind]]
    test_records = list(csv.DictReader((DATA_DIR / "synthetic_telemetry.csv").open(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(9, 5))
    for sid in selected:
        records = [r for r in test_records if int(r["scenario_id"]) == sid]
        if not records: continue
        info = scenario_info[sid]
        service = info["target_service"] if info["scenario_type"] == "attack" else "gateway"
        service_rows = [r for r in records if r["service"] == service]
        ax.plot([int(r["window"]) for r in service_rows], [float(r["request_rate"]) for r in service_rows], label=f"{info['scenario_type']}: {service}")
        if records[0]["scenario_type"] == "attack":
            ax.axvspan(int(info["event_start"]), int(info["event_end"]), alpha=.12, color="red", label="simulated attack window")
    ax.set(title="Synthetic request-rate examples (red bands mark simulated attack windows)", xlabel="Window", ylabel="Requests / second"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "traffic_over_time.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6)); angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(cfg["services"]), endpoint=False)
    positions = {service: (float(np.cos(angle)), float(np.sin(angle))) for service, angle in zip(cfg["services"], angles)}
    for edge in graph["edges"]:
        ax.annotate("", xy=positions[edge[1]], xytext=positions[edge[0]], arrowprops={"arrowstyle": "->", "color": "#64748b", "connectionstyle": "arc3,rad=0.15", "shrinkA": 24, "shrinkB": 24})
    ax.scatter([positions[s][0] for s in cfg["services"]], [positions[s][1] for s in cfg["services"]], s=1700, color="#dbeafe", edgecolor="#2563eb", zorder=2)
    for service in cfg["services"]: ax.text(*positions[service], service, ha="center", va="center", fontsize=9, zorder=3)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("Configured synthetic microservice call graph")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "service_graph.png", dpi=160); plt.close(fig)
    write_dashboard(result)
    print(f"Evaluation complete. Actual run metrics and plots saved in {OUTPUT_DIR}")


if __name__ == "__main__": main()
