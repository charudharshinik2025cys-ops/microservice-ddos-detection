"""Serve a loopback-only dashboard that injects synthetic telemetry, not traffic."""
import csv
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pickle
import urllib.parse

import numpy as np

from .common import CLASSES, FEATURES, OUTPUT_DIR, DATA_DIR, ensure_dirs, load_config

HOST, PORT = "127.0.0.1", 8765
EVENT_NUMBER = 0


def simulate(target_service):
    """Make a synthetic time window and classify every service with the RF model."""
    global EVENT_NUMBER
    cfg = load_config()
    services = cfg["services"]
    if target_service not in services:
        raise ValueError("Choose one of the listed services.")
    if not (OUTPUT_DIR / "random_forest.pkl").exists() or not (OUTPUT_DIR / "scaler.pkl").exists():
        raise RuntimeError("Models are missing. First run: python -m src.run_pipeline")
    graph = json.loads((DATA_DIR / "service_graph.json").read_text(encoding="utf-8"))
    index = {service: i for i, service in enumerate(services)}
    children = {service: [] for service in services}
    for source, destination in graph["edges"]:
        children[source].append(destination)
    target_index = index[target_service]
    affected = set()
    frontier = [(target_service, 1)]
    while frontier:
        service, depth = frontier.pop(0)
        if depth > 2:
            continue
        for child in children[service]:
            if child != target_service and child not in affected:
                affected.add(child)
                frontier.append((child, depth + 1))

    EVENT_NUMBER += 1
    rng = np.random.default_rng(cfg["seed"] + EVENT_NUMBER)
    feature_matrix = np.zeros((len(services), len(FEATURES)), dtype=float)
    truth = []
    for i, service in enumerate(services):
        role_load = 1.5 if service == "gateway" else 1.0
        rate = 90 * role_load * rng.lognormal(0, .08)
        latency, errors, timeouts, cpu, memory = 75.0, .012, .004, 28.0, 42.0
        label = "normal"
        if service == target_service:
            label = "directly_targeted"
            rate *= rng.uniform(5.5, 7.5)
            latency += rng.uniform(170, 250)
            errors += rng.uniform(.20, .38)
            timeouts += rng.uniform(.08, .20)
            cpu += rng.uniform(42, 62)
        elif service in affected:
            label = "indirectly_affected"
            rate *= rng.uniform(1.2, 1.8)
            latency += rng.uniform(35, 95)
            errors += rng.uniform(.04, .16)
            timeouts += rng.uniform(.015, .08)
            cpu += rng.uniform(8, 25)
        latency = max(5, latency + rng.normal(0, 7))
        cpu = float(np.clip(cpu + rng.normal(0, 4), 2, 99))
        memory = float(np.clip(memory + .12 * (cpu - 28) + rng.normal(0, 2), 10, 99))
        feature_matrix[i] = [rate, latency, min(errors, .99), min(timeouts, .99), cpu, memory]
        truth.append(label)

    with (OUTPUT_DIR / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (OUTPUT_DIR / "random_forest.pkl").open("rb") as handle:
        model = pickle.load(handle)
    prediction_ids = model.predict(scaler.transform(feature_matrix))
    predictions = [CLASSES[int(value)] for value in prediction_ids]
    flagged = [service for service, label in zip(services, predictions) if label == "directly_targeted"]
    indirect = [service for service, label in zip(services, predictions) if label == "indirectly_affected"]
    rows = [{"service": service, **{feature: float(value) for feature, value in zip(FEATURES, feature_matrix[i])},
             "known_simulation_label": truth[i], "model_prediction": predictions[i]}
            for i, service in enumerate(services)]
    result = {"detected": bool(flagged), "target_service": target_service, "flagged_services": flagged,
              "affected_services": indirect,
              "services": [{"service": s, "prediction": p} for s, p in zip(services, predictions)],
              "time": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
              "notice": "Synthetic local simulation; no attack traffic was sent.",
              # The dashboard needs these values to redraw its live request-rate chart.
              "telemetry": rows}
    ensure_dirs()
    (OUTPUT_DIR / "live_demo_latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    history = OUTPUT_DIR / "live_demo_history.csv"
    with history.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_number", "time", "target_service", "detected", "flagged_services"])
        if history.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({"event_number": EVENT_NUMBER, "time": result["time"], "target_service": target_service,
                         "detected": result["detected"], "flagged_services": ";".join(flagged)})
    return result


class DashboardHandler(BaseHTTPRequestHandler):
    def send_bytes(self, code, content_type, content):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path not in ("/", "/dashboard.html"):
            self.send_error(404)
            return
        path = OUTPUT_DIR / "dashboard.html"
        if not path.exists():
            self.send_error(404, "Dashboard missing. Run python -m src.run_pipeline first.")
            return
        self.send_bytes(200, "text/html; charset=utf-8", path.read_bytes())

    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        if route == "/reset":
            (OUTPUT_DIR / "live_demo_latest.json").unlink(missing_ok=True)
            self.send_bytes(200, "application/json", b'{"ok":true}')
            return
        if route != "/simulate":
            self.send_error(404)
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request = json.loads(body or b"{}")
            result = simulate(str(request.get("target_service", "")))
            self.send_bytes(200, "application/json; charset=utf-8", json.dumps(result).encode("utf-8"))
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            content = json.dumps({"error": str(error)}).encode("utf-8")
            self.send_bytes(400, "application/json; charset=utf-8", content)

    def log_message(self, fmt, *args):
        print("[local dashboard] " + fmt % args)


def main():
    ensure_dirs()
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Synthetic demo dashboard running at http://{HOST}:{PORT}")
    print("Only this computer can connect. Press Ctrl+C to stop the demo.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local demo server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
