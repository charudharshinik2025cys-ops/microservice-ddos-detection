# Microservice-Level DDoS Detection and Target Localization Using Graph Convolutional Networks

This is a beginner-friendly, local-only academic simulation. It generates artificial microservice traffic, trains a graph convolutional network (GCN), and compares it with a Random Forest and a transparent threshold rule. **All data and reported results are synthetic. They do not demonstrate effectiveness against real attacks.** The project only simulates telemetry; it never sends traffic over a network.

## What it does

- Builds a configurable directed service-call graph and produces normal, legitimate flash-sale-like spike, and simulated request-flood scenarios.
- Generates per-service request rate, average latency, error rate, timeout rate, CPU, and memory for every time window.
- Labels each service-window as `normal`, `directly_targeted`, or `indirectly_affected`.
- Trains a two-layer GCN on service features and a normalized version of the call graph. A small NumPy implementation keeps the training CPU-only and avoids a deep-learning framework.
- Compares a Random Forest using the same features, a threshold rule, and an ablation where the GCN's graph edges are removed.
- Splits by complete scenario IDs (70% train, 30% test), so neighboring windows from a scenario cannot leak across the split. The feature scaler is fitted on training scenarios only.
- Writes per-class precision, recall, F1, confusion matrices, normal/spike false-positive rates, target localization, and detection delay to files. The delay is the first attack window where the true target is predicted directly targeted, minus the attack start; misses are excluded from the mean delay.

## Folder layout

```text
.
├── config.json                 # seed, topology, scenario count, training settings
├── data/                       # generated CSV and graph/scenario JSON files
├── outputs/                    # trained model artifacts, metrics, and plots
├── src/
│   ├── common.py
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── dashboard_template.html
│   ├── live_demo.py
│   └── run_pipeline.py
├── .vscode/launch.json
└── requirements.txt
```

The generated CSV `data/synthetic_telemetry.csv` has one row per scenario, time window, and service. `data/scenarios.json` records scenario types, simulated attack intervals, and targets. `data/service_graph.json` records the topology. All paths are relative to this project folder.

## Run in VS Code on Windows

1. Open this folder in VS Code (**File → Open Folder**).
2. Open **Terminal → New Terminal**. The commands below use PowerShell.
3. Create and activate a virtual environment, then install the small CPU-compatible dependencies:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. In VS Code, choose **Python: Select Interpreter** from the Command Palette and select `.venv`.

4. Run the stages individually:

   ```powershell
   python -m src.generate_data
   python -m src.train
   python -m src.evaluate
   ```

5. Or run the complete pipeline with one command:

   ```powershell
   python -m src.run_pipeline
   ```

6. To demonstrate a **local synthetic attack** using the trained Random Forest, start the dashboard server in the VS Code terminal:

   ```powershell
   python -m src.live_demo
   ```

   Open `http://127.0.0.1:8765` in your browser. Choose a service and click **Simulate attack**. The project generates one synthetic service-telemetry window, runs the detector, and displays the services it flags. This button never sends attack requests to a service or the internet. Keep the terminal open while presenting; press `Ctrl+C` there to stop the dashboard. Each run is recorded in `outputs/live_demo_latest.json` and `outputs/live_demo_history.csv`.

   For a presentation, you can place the two commands in separate VS Code terminals: first run `python -m src.run_pipeline` and wait until it finishes, then run `python -m src.live_demo` and leave that terminal open while using the browser dashboard.

You can also use **Run and Debug → Run complete synthetic pipeline** with `.vscode/launch.json`. To change the default reproducible seed, edit `seed` in `config.json`; to override it for data generation only, run `python -m src.generate_data --seed 7` (then train and evaluate).

If the `py` launcher is unavailable, install Python 3.10 or newer from python.org and use `python -m venv .venv` instead. If a dependency install fails, check that the virtual environment is active and that pip can reach PyPI; the required packages are NumPy, scikit-learn, and Matplotlib, all commonly available as Windows wheels.

## Outputs

After a successful run, `outputs/` contains `gcn_weights.npz`, `random_forest.pkl`, `scaler.pkl`, and the scenario split; `evaluation.json` and `evaluation_summary.csv` hold the measured results from that run. Use the local dashboard at `http://127.0.0.1:8765` for the interactive demonstration. After each simulation, its service map highlights the selected target and the services the model flags, and its request-rate chart redraws from the new simulated telemetry. The static `outputs/dashboard.html` preview can also be opened directly, but its simulate button works only through the local server. `confusion_matrices.png`, `service_graph.png`, and `traffic_over_time.png` are available individually. The evaluation numbers are calculated from the generated test scenarios, never pre-filled metrics.

`gcn_without_edges` is the requested graph ablation. It uses the trained GCN weights but replaces graph propagation with self-only adjacency at evaluation time. All evaluation output is for this synthetic simulation and should be interpreted only as a learning exercise.
