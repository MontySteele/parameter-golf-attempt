"""
Batch experiment runner for parameter-golf.
 
Usage:
    python run_experiments.py                          # run ALL experiments
    python run_experiments.py 6bit_7Lx2 6bit_9Lx2     # run specific ones
    python run_experiments.py --list                   # show available experiments
    python run_experiments.py --results                # show past results
 
Reads experiment configs from experiments.json, runs each one sequentially,
and collects results into results.json.
"""
 
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
 
CONFIG_FILE = "experiments.json"
RESULTS_FILE = "results.json"
 
 
def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)
 
 
def load_results():
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []
 
 
def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
 
 
def list_experiments(config):
    print("\nAvailable experiments:")
    print("-" * 80)
    for name, exp in config["experiments"].items():
        desc = exp.get("description", "")
        script = exp.get("script", "train_gpt_6bit_recurrence_adapter.py")
        print(f"  {name:<30s} {script:<25s} {desc}")
    print()
 
 
def show_results():
    results = load_results()
    if not results:
        print("\nNo results yet.\n")
        return
 
    print("\n" + "=" * 100)
    print(f"{'Experiment':<30s} {'val_bpb':>10s} {'post_quant':>12s} {'steps':>8s} "
          f"{'params':>10s} {'size_mb':>10s} {'date':>20s}")
    print("-" * 100)
 
    # Sort by val_bpb (best first)
    for r in sorted(results, key=lambda x: x.get("val_bpb", 999)):
        print(f"{r.get('name', '?'):<30s} "
              f"{r.get('val_bpb', 0):>10.4f} "
              f"{r.get('post_quant_bpb', 0):>12.4f} "
              f"{r.get('steps', 0):>8d} "
              f"{r.get('params', 0):>10,d} "
              f"{r.get('compressed_mb', 0):>10.2f} "
              f"{r.get('timestamp', ''):>20s}")
    print("=" * 100)
    print()
 
 
def parse_output(output_text):
    """Extract key metrics from training script stdout."""
    metrics = {}
 
    # model_params
    m = re.search(r"model_params:(\d+)", output_text)
    if m:
        metrics["params"] = int(m.group(1))
 
    # Find the last val_loss/val_bpb line before "stopping_early" or "peak memory"
    # This captures the end-of-training validation
    val_matches = re.findall(
        r"val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:([\d.]+)ms\s+step_avg:([\d.]+)ms",
        output_text
    )
    if val_matches:
        last = val_matches[-1]
        metrics["val_loss"] = float(last[0])
        metrics["val_bpb"] = float(last[1])
        metrics["train_time_ms"] = float(last[2])
        metrics["step_avg_ms"] = float(last[3])
 
    # Steps completed
    m = re.search(r"step:(\d+)/\d+\s+val_loss", output_text)
    if m:
        metrics["steps"] = int(m.group(1))
 
    # Peak memory
    m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB", output_text)
    if m:
        metrics["peak_memory_mib"] = int(m.group(1))
 
    # Compressed size
    m = re.search(r"Total submission size int8\+zlib:\s*(\d+)\s*bytes", output_text)
    if m:
        metrics["compressed_bytes"] = int(m.group(1))
        metrics["compressed_mb"] = int(m.group(1)) / 1e6
 
    # Post-quant roundtrip
    m = re.search(r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)", output_text)
    if m:
        metrics["post_quant_loss"] = float(m.group(1))
        metrics["post_quant_bpb"] = float(m.group(2))
 
    return metrics
 
 
def run_experiment(name, config):
    """Run a single experiment and return metrics."""
    defaults = config.get("defaults", {})
    exp = config["experiments"][name]
 
    # Merge defaults with experiment-specific overrides
    env = os.environ.copy()
    for key, val in defaults.items():
        env[key] = str(val)
    for key, val in exp.items():
        if key not in ("description", "script"):
            env[key] = str(val)
 
    # Set RUN_ID to experiment name
    env["RUN_ID"] = name
 
    script = exp.get("script", "train_gpt.py")
    description = exp.get("description", "")
 
    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print(f"  {description}")
    print(f"  Script: {script}")
 
    # Show non-default settings
    overrides = {k: v for k, v in exp.items() if k not in ("description", "script")}
    if overrides:
        print(f"  Overrides: {overrides}")
    print("=" * 80 + "\n")
 
    start = time.time()
 
    # Run the training script, streaming output to console
    proc = subprocess.Popen(
        [sys.executable, script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
 
    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
 
    proc.wait()
    elapsed = time.time() - start
    output_text = "".join(output_lines)
 
    if proc.returncode != 0:
        print(f"\n*** EXPERIMENT FAILED: {name} (exit code {proc.returncode}) ***\n")
        return None
 
    # Parse metrics from output
    metrics = parse_output(output_text)
    metrics["name"] = name
    metrics["description"] = description
    metrics["script"] = script
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    metrics["overrides"] = overrides
 
    print(f"\n--- {name} complete in {elapsed:.0f}s ---")
    if "val_bpb" in metrics:
        print(f"    val_bpb: {metrics['val_bpb']:.4f}")
    if "post_quant_bpb" in metrics:
        print(f"    post_quant_bpb: {metrics['post_quant_bpb']:.4f}")
    if "compressed_mb" in metrics:
        print(f"    compressed: {metrics['compressed_mb']:.2f} MB")
    print()
 
    return metrics
 
 
def main():
    args = sys.argv[1:]
 
    if "--list" in args:
        list_experiments(load_config())
        return
 
    if "--results" in args:
        show_results()
        return
 
    config = load_config()
    results = load_results()
 
    # Determine which experiments to run
    if args:
        experiment_names = [a for a in args if not a.startswith("--")]
        # Validate names
        for name in experiment_names:
            if name not in config["experiments"]:
                print(f"Unknown experiment: {name}")
                print(f"Available: {', '.join(config['experiments'].keys())}")
                return
    else:
        experiment_names = list(config["experiments"].keys())
 
    print(f"\nWill run {len(experiment_names)} experiment(s): {', '.join(experiment_names)}")
    print(f"Estimated time: ~{len(experiment_names) * 14} minutes (10 min train + ~4 min eval each)\n")
 
    for name in experiment_names:
        metrics = run_experiment(name, config)
        if metrics:
            results.append(metrics)
            save_results(results)
 
    # Show summary
    print("\n\n")
    show_results()
 
 
if __name__ == "__main__":
    main()