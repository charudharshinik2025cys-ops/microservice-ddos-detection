"""Generate fully synthetic service telemetry; no network traffic is created."""
import argparse
import csv
import json

import numpy as np

from .common import DATA_DIR, ensure_dirs, load_config


def generate(seed=None):
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    services, edges = cfg["services"], cfg["edges"]
    names = {name: i for i, name in enumerate(services)}
    children = {i: [] for i in range(len(services))}
    for source, target in edges:
        children[names[source]].append(names[target])
    rows, scenario_meta = [], []
    for sid in range(cfg["scenarios"]):
        kind = ["normal", "spike", "attack"][sid % 3]
        target = int(rng.integers(0, len(services))) if kind == "attack" else -1
        start = int(rng.integers(5, cfg["windows_per_scenario"] - 5)) if kind != "normal" else -1
        duration = int(rng.integers(3, 7))
        scenario_meta.append({"scenario_id": sid, "scenario_type": kind,
                              "target_service": services[target] if target >= 0 else "",
                              "event_start": start, "event_end": start + duration if start >= 0 else -1})
        for t in range(cfg["windows_per_scenario"]):
            demand = rng.uniform(0.82, 1.18) * (1 + 0.10 * np.sin(2 * np.pi * t / 12))
            attacked = kind == "attack" and start <= t < start + duration
            spike = kind == "spike" and start <= t < start + duration
            event_strength = rng.uniform(1.8, 2.5) if spike else 1.0
            direct = {target} if attacked else set()
            indirect = set()
            if attacked:
                frontier = [(target, 1)]
                while frontier:
                    node, depth = frontier.pop(0)
                    if depth <= 2:
                        for child in children[node]:
                            if child not in direct:
                                indirect.add(child)
                                frontier.append((child, depth + 1))
            for i, service in enumerate(services):
                role_load = 1.5 if service == "gateway" else 1.0
                rate = 90 * role_load * demand * rng.lognormal(0, 0.08)
                latency, errors, timeouts = 75.0, 0.012, 0.004
                cpu, memory = 28.0, 42.0
                label = "normal"
                # Flash-sale traffic is legitimate and has no positive class label.
                if spike:
                    rate *= event_strength
                    latency += 13 * event_strength
                    cpu += 10 * event_strength
                if i in direct:
                    label = "directly_targeted"
                    rate *= rng.uniform(5.5, 7.5)
                    latency += rng.uniform(170, 250)
                    errors += rng.uniform(0.20, 0.38)
                    timeouts += rng.uniform(0.08, 0.20)
                    cpu += rng.uniform(42, 62)
                elif i in indirect:
                    label = "indirectly_affected"
                    rate *= rng.uniform(1.2, 1.8)
                    latency += rng.uniform(35, 95)
                    errors += rng.uniform(0.04, 0.16)
                    timeouts += rng.uniform(0.015, 0.08)
                    cpu += rng.uniform(8, 25)
                latency = max(5, latency + rng.normal(0, 7))
                cpu = float(np.clip(cpu + rng.normal(0, 4), 2, 99))
                memory = float(np.clip(memory + 0.12 * (cpu - 28) + rng.normal(0, 2), 10, 99))
                rows.append({"scenario_id": sid, "scenario_type": kind, "window": t,
                             "service": service, "request_rate": round(rate, 3),
                             "avg_latency_ms": round(latency, 3), "error_rate": round(min(errors, .99), 5),
                             "timeout_rate": round(min(timeouts, .99), 5), "cpu_percent": round(cpu, 3),
                             "memory_percent": round(memory, 3), "label": label,
                             "is_attack_window": int(attacked),
                             "target_service": services[target] if target >= 0 else ""})
    ensure_dirs()
    with (DATA_DIR / "synthetic_telemetry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (DATA_DIR / "scenarios.json").write_text(json.dumps(scenario_meta, indent=2), encoding="utf-8")
    (DATA_DIR / "service_graph.json").write_text(json.dumps({"services": services, "edges": edges}, indent=2), encoding="utf-8")
    print(f"Generated {len(rows)} synthetic service-window rows in {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Override seed in config.json")
    generate(parser.parse_args().seed)
