"""
Batch experiment runner for parameter-golf.

Usage:
    python run_experiments.py                          # run ALL experiments
    python run_experiments.py 6bit_7Lx2 6bit_9Lx2     # run specific ones
    python run_experiments.py --list                   # show available experiments
    python run_experiments.py --results                # show past results
    python run_experiments.py --resume                 # resume from last interrupted queue

Reads experiment configs from experiments.json, runs each one sequentially,
and collects results into results.json.

Supports graceful Ctrl+C interruption:
  - Ctrl+C while an experiment is running: the subprocess is terminated, partial
    metrics are saved, and the remaining queue is written to run_progress.json.
  - Use --resume to pick up where you left off.

Mid-run progress is logged to logs/<experiment_name>.jsonl so you can inspect
training curves even for incomplete runs.
"""

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Keep-awake: prevent Windows from sleeping during long experiment runs
# ---------------------------------------------------------------------------

@contextmanager
def keep_awake():
    """
    Context manager that prevents the OS from sleeping while experiments run.

    On Windows, calls SetThreadExecutionState to tell the OS "don't sleep".
    On other platforms, this is a no-op.
    """
    if platform.system() != "Windows":
        yield
        return

    # Windows execution state flags
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002  # also keeps display on (optional but helpful)

    try:
        prev = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        if prev == 0:
            print("(warning: could not set keep-awake; check power settings)")
        else:
            print("(keep-awake enabled — system will not sleep during experiments)")
        yield
    finally:
        # Restore normal sleep behavior
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


CONFIG_FILE = "experiments.json"
RESULTS_FILE = "results.json"
PROGRESS_FILE = "run_progress.json"
LOGS_DIR = Path("logs")
MAX_CONSECUTIVE_FAILURES = 3  # stop queue after this many failures in a row


# ---------------------------------------------------------------------------
# Live progress bar
# ---------------------------------------------------------------------------

class ProgressBar:
    """
    In-place terminal progress bar for a running experiment.

    Renders a single updating line at the bottom of the terminal:

      6bit_7Lx2_ada | loading model | setup 0:12
      6bit_7Lx2_ada | step 450 | loss 2.534 | bpb 1.496 | [████████░░░░] 4:32/10:00

    A background thread redraws the bar every second so the elapsed timer
    stays live even when the subprocess is silent (e.g. during model init or
    data loading).

    The training timer only starts when the @@PHASE:training marker is received
    from the subprocess, so the 10-minute wallclock budget reflects actual
    training time, not setup overhead.

    Non-progress log lines are printed *above* the bar by first clearing it,
    printing the line, then redrawing.  Falls back to no-op when stdout is
    not a tty (e.g. piped to a file).
    """

    # Block characters for the bar fill
    FULL  = "\u2588"   # █
    EMPTY = "\u2591"   # ░

    # Human-readable labels for each phase
    PHASE_LABELS = {
        "init":         "starting",
        "tokenizer":    "loading tokenizer",
        "model":        "building model",
        "resume":       "loading checkpoint",
        "dataloader":   "loading data",
        "warmup":       "warmup",
        "training":     "training",
        "train":        "training",
        "eval":         "validating",
        "quantize":     "quantizing",
        "quant":        "quantizing",
        "interrupted":  "saving checkpoint",
    }

    def __init__(self, name: str, wallclock_budget: float):
        self.name = name
        self.wallclock_budget = wallclock_budget  # seconds (training only)
        self.start = time.time()         # wall-clock start of the whole run
        self.train_start = None          # set when @@PHASE:training is seen
        self.step = 0
        self.total_steps = 0
        self.train_loss = None
        self.val_bpb = None
        self.phase = "init"
        self.train_frozen_elapsed = None  # set when training ends (eval/quant phase)
        self._enabled = sys.stdout.isatty()
        self._last_bar = ""
        # Lock protects all terminal writes so the background thread and the
        # main thread (print_line / clear) never interleave partial output.
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ticker: threading.Thread | None = None
        if self._enabled:
            self._start_ticker()

    def _start_ticker(self):
        """Spawn a daemon thread that redraws the bar once per second."""
        def _run():
            while not self._stop_event.wait(timeout=1.0):
                with self._lock:
                    if not self._stop_event.is_set():
                        self._draw_bar()
        self._ticker = threading.Thread(target=_run, daemon=True)
        self._ticker.start()

    # -- public API ----------------------------------------------------------

    def update_from_line(self, line: str):
        """Parse a stdout line and update internal state."""
        stripped = line.strip()

        # @@PHASE markers emitted by the training scripts
        m = re.match(r"@@PHASE:(\w+)", stripped)
        if m:
            new_phase = m.group(1)
            # Freeze the training elapsed time when we leave the training phase
            # so validation/quantization time doesn't inflate the training timer.
            if (self.phase in ("training", "train")
                    and new_phase not in ("training", "train")
                    and self.train_start is not None
                    and self.train_frozen_elapsed is None):
                self.train_frozen_elapsed = time.time() - self.train_start
            self.phase = new_phase
            if new_phase == "training" and self.train_start is None:
                self.train_start = time.time()
            return

        # Training step
        m = re.match(
            r"step:(\d+)/(\d+)\s+train_loss:([\d.]+)",
            stripped,
        )
        if m:
            self.step = int(m.group(1))
            self.total_steps = int(m.group(2))
            self.train_loss = float(m.group(3))
            self.phase = "train"
            if self.train_start is None:
                self.train_start = time.time()
            return

        # Validation step (may also contain train_time etc.)
        m = re.match(
            r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)",
            stripped,
        )
        if m:
            self.step = int(m.group(1))
            self.total_steps = int(m.group(2))
            self.val_bpb = float(m.group(4))
            self.phase = "eval"
            return

        # Warmup step
        m = re.match(r"warmup_step:(\d+)/(\d+)", stripped)
        if m:
            self.phase = "warmup"
            return

    def print_line(self, line: str):
        """Print a log line above the progress bar (suppress @@PHASE lines)."""
        if line.strip().startswith("@@PHASE:"):
            return  # don't print the marker itself
        if not self._enabled:
            print(line, end="")
            return
        with self._lock:
            self._clear_bar()
            print(line, end="")
            self._draw_bar()

    def tick(self):
        """Redraw the bar (call after update_from_line)."""
        if self._enabled:
            with self._lock:
                self._draw_bar()

    def clear(self):
        """Stop the ticker and remove the bar from the terminal."""
        self._stop_event.set()
        if self._ticker is not None:
            self._ticker.join(timeout=2)
        if self._enabled:
            with self._lock:
                self._clear_bar()

    # -- internals (caller must hold self._lock) -----------------------------

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    def _training_active(self) -> bool:
        """True once the training loop has started."""
        return self.train_start is not None

    def _train_elapsed(self) -> float:
        """Seconds of actual training (frozen once training ends)."""
        if self.train_frozen_elapsed is not None:
            return self.train_frozen_elapsed
        if self.train_start is None:
            return 0.0
        return time.time() - self.train_start

    def _build_bar(self) -> str:
        try:
            cols = shutil.get_terminal_size((80, 24)).columns
        except Exception:
            cols = 80

        segments = [self.name]

        # Phase label
        phase_label = self.PHASE_LABELS.get(self.phase, self.phase)

        in_training = self.phase in ("training", "train")
        post_training = self._training_active() and not in_training  # eval, quantize, etc.

        if post_training:
            # Post-training phase: show phase label, final metrics, frozen training time
            segments.append(phase_label)
            if self.val_bpb is not None:
                segments.append(f"bpb {self.val_bpb:.4f}")
            train_elapsed = self._train_elapsed()
            segments.append(f"trained {self._fmt_time(train_elapsed)}")
            frac = 1.0  # bar is full — training is done
        elif self._training_active():
            # During training: show step, metrics, live training timer
            if self.step > 0:
                segments.append(f"step {self.step}")
            else:
                segments.append(phase_label)

            if self.train_loss is not None:
                segments.append(f"loss {self.train_loss:.3f}")
            if self.val_bpb is not None:
                segments.append(f"bpb {self.val_bpb:.4f}")

            train_elapsed = self._train_elapsed()
            time_str = f"{self._fmt_time(train_elapsed)}/{self._fmt_time(self.wallclock_budget)}"
            segments.append(time_str)

            frac = min(train_elapsed / self.wallclock_budget, 1.0) if self.wallclock_budget > 0 else 0
        else:
            # Setup phase: show phase label and setup timer (no progress bar fill)
            segments.append(phase_label)
            setup_elapsed = time.time() - self.start
            segments.append(f"setup {self._fmt_time(setup_elapsed)}")
            frac = 0.0

        info = " | ".join(segments)

        # Calculate how wide the bar portion can be
        bar_overhead = len(info) + 6  # "  [...]  "
        bar_width = max(cols - bar_overhead, 10)

        filled = int(frac * bar_width)
        bar = self.FULL * filled + self.EMPTY * (bar_width - filled)

        return f" {info} [{bar}]"

    def _draw_bar(self):
        bar = self._build_bar()
        self._last_bar = bar
        sys.stdout.write(f"\r{bar}")
        sys.stdout.flush()

    def _clear_bar(self):
        if self._last_bar:
            sys.stdout.write("\r" + " " * len(self._last_bar) + "\r")
            sys.stdout.flush()
            self._last_bar = ""


# ---------------------------------------------------------------------------
# Config / results / progress persistence
# ---------------------------------------------------------------------------

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


def load_progress():
    """Load the queue progress file (tracks which experiments remain)."""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
        if not data.get("remaining"):
            return None
        return data
    return None


def save_progress(queue_remaining, current_experiment=None):
    """Save queue state so we can resume after interruption."""
    progress = {
        "remaining": queue_remaining,
        "current": current_experiment,
        "timestamp": datetime.now().isoformat(),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def clear_progress():
    """Clear the progress file when the queue finishes cleanly."""
    try:
        if Path(PROGRESS_FILE).exists():
            with open(PROGRESS_FILE, "w") as f:
                json.dump({"remaining": [], "finished": True, "timestamp": datetime.now().isoformat()}, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_output(output_text):
    """Extract key metrics from training script stdout."""
    metrics = {}

    m = re.search(r"model_params:(\d+)", output_text)
    if m:
        metrics["params"] = int(m.group(1))

    val_matches = re.findall(
        r"val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:([\d.]+)ms\s+step_avg:([\d.]+)ms",
        output_text,
    )
    if val_matches:
        last = val_matches[-1]
        metrics["val_loss"] = float(last[0])
        metrics["val_bpb"] = float(last[1])
        metrics["train_time_ms"] = float(last[2])
        metrics["step_avg_ms"] = float(last[3])

    step_matches = re.findall(r"step:(\d+)/(\d+)", output_text)
    if step_matches:
        metrics["steps"] = max(int(s[0]) for s in step_matches)
        metrics["total_steps"] = int(step_matches[-1][1])

    m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB", output_text)
    if m:
        metrics["peak_memory_mib"] = int(m.group(1))

    m = re.search(r"Total submission size int8\+zlib:\s*(\d+)\s*bytes", output_text)
    if m:
        metrics["compressed_bytes"] = int(m.group(1))
        metrics["compressed_mb"] = int(m.group(1)) / 1e6

    m = re.search(r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)", output_text)
    if m:
        metrics["post_quant_loss"] = float(m.group(1))
        metrics["post_quant_bpb"] = float(m.group(2))

    return metrics


def parse_incremental_line(line):
    """Parse a single output line for the .jsonl progress log."""
    progress = {}

    m = re.match(
        r"step:(\d+)/(\d+)\s+train_loss:([\d.]+)\s+train_time:([\d.]+)ms\s+step_avg:([\d.]+)ms",
        line.strip(),
    )
    if m:
        progress["type"] = "train_step"
        progress["step"] = int(m.group(1))
        progress["total_steps"] = int(m.group(2))
        progress["train_loss"] = float(m.group(3))
        progress["train_time_ms"] = float(m.group(4))
        progress["step_avg_ms"] = float(m.group(5))
        return progress

    m = re.match(
        r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)",
        line.strip(),
    )
    if m:
        progress["type"] = "val_step"
        progress["step"] = int(m.group(1))
        progress["total_steps"] = int(m.group(2))
        progress["val_loss"] = float(m.group(3))
        progress["val_bpb"] = float(m.group(4))
        return progress

    return None


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(name, config):
    """
    Run a single experiment and return metrics.

    Raises KeyboardInterrupt if Ctrl+C is pressed — the caller handles
    saving partial results and queue progress.
    """
    defaults = config.get("defaults", {})
    exp = config["experiments"][name]

    # Merge defaults with experiment-specific overrides
    env = os.environ.copy()
    for key, val in defaults.items():
        env[key] = str(val)
    for key, val in exp.items():
        if key not in ("description", "script"):
            env[key] = str(val)

    env["RUN_ID"] = name

    # Check for a mid-training checkpoint from a previous interrupted run
    checkpoint_file = LOGS_DIR / f"{name}_checkpoint.pt"
    if checkpoint_file.exists():
        env["RESUME_CHECKPOINT"] = str(checkpoint_file)

    script = exp.get("script", "train_gpt.py")
    description = exp.get("description", "")

    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print(f"  {description}")
    print(f"  Script: {script}")

    overrides = {k: v for k, v in exp.items() if k not in ("description", "script")}
    if overrides:
        print(f"  Overrides: {overrides}")
    if checkpoint_file.exists():
        print(f"  Resuming from checkpoint: {checkpoint_file}")
    print("=" * 80 + "\n")

    start = time.time()

    # Wallclock budget for the progress bar (from env or defaults)
    wallclock = float(env.get("MAX_WALLCLOCK_SECONDS", 600))

    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"{name}.txt"
    progress_log = LOGS_DIR / f"{name}.jsonl"

    # Force unbuffered stdout in the child so @@PHASE markers and step logs
    # appear immediately rather than sitting in an 8KB pipe buffer.
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    bar = ProgressBar(name, wallclock)
    output_lines = []
    last_progress = None
    was_interrupted = False

    try:
        with open(log_file, "w") as lf, open(progress_log, "w") as pf:
            for line in proc.stdout:
                output_lines.append(line)

                # Write to full log file (always)
                lf.write(line)
                lf.flush()

                # Update progress bar state from this line
                bar.update_from_line(line)

                # Print the log line *above* the bar, then redraw the bar
                bar.print_line(line)
                bar.tick()

                # Append to incremental .jsonl log
                progress = parse_incremental_line(line)
                if progress:
                    progress["timestamp"] = time.time()
                    progress["elapsed_s"] = round(time.time() - start, 1)
                    pf.write(json.dumps(progress) + "\n")
                    pf.flush()
                    last_progress = progress
    except KeyboardInterrupt:
        was_interrupted = True
        bar.clear()
        print("\n\n*** Ctrl+C — saving checkpoint, please wait... ***\n")
        try:
            # Give the training script time to save its checkpoint (model + optimizer state)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("*** Checkpoint save timed out, killing process ***")
            proc.kill()
            proc.wait()

    # Clear the bar before printing final summary
    bar.clear()

    if not was_interrupted:
        proc.wait()

    elapsed = time.time() - start
    output_text = "".join(output_lines)

    if proc.returncode != 0 and not was_interrupted:
        print(f"\n*** EXPERIMENT FAILED: {name} (exit code {proc.returncode}) ***\n")
        return None

    # Parse metrics from output (works even for partial runs)
    metrics = parse_output(output_text)
    metrics["name"] = name
    metrics["description"] = description
    metrics["script"] = script
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    metrics["overrides"] = overrides

    if was_interrupted:
        metrics["status"] = "interrupted"
        if "val_bpb" not in metrics and last_progress and "val_bpb" in last_progress:
            metrics["val_bpb"] = last_progress["val_bpb"]
            metrics["val_loss"] = last_progress["val_loss"]
        if "steps" not in metrics and last_progress and "step" in last_progress:
            metrics["steps"] = last_progress["step"]
        print(f"\n--- {name} INTERRUPTED after {elapsed:.0f}s ---")
    else:
        metrics["status"] = "completed"
        print(f"\n--- {name} complete in {elapsed:.0f}s ---")

    if "val_bpb" in metrics:
        print(f"    val_bpb: {metrics['val_bpb']:.4f}")
    if "post_quant_bpb" in metrics:
        print(f"    post_quant_bpb: {metrics['post_quant_bpb']:.4f}")
    if "compressed_mb" in metrics:
        print(f"    compressed: {metrics['compressed_mb']:.2f} MB")
    if "steps" in metrics:
        print(f"    steps completed: {metrics['steps']}")
    print()

    # Back up model weights so the next experiment doesn't overwrite them
    model_file = Path("final_model.pt")
    if model_file.exists():
        backup = LOGS_DIR / f"{name}_model.pt"
        try:
            shutil.copy2(model_file, backup)
            metrics["model_checkpoint"] = str(backup)
            print(f"    model weights backed up to: {backup}")
        except Exception as e:
            print(f"    (warning: could not backup model weights: {e})")

    quant_file = Path("final_model.int8.ptz")
    if quant_file.exists():
        backup = LOGS_DIR / f"{name}_model.int8.ptz"
        try:
            shutil.copy2(quant_file, backup)
            metrics["quant_checkpoint"] = str(backup)
        except Exception:
            pass

    if was_interrupted:
        raise KeyboardInterrupt

    # Successful completion — remove any stale checkpoint from a previous interrupted run
    checkpoint_file = LOGS_DIR / f"{name}_checkpoint.pt"
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            print(f"    cleaned up checkpoint: {checkpoint_file}")
        except OSError:
            pass

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    if "--resume" in args:
        progress = load_progress()
        if progress and progress.get("remaining"):
            experiment_names = progress["remaining"]
            print(f"\nResuming from interrupted queue (saved {progress.get('timestamp', '?')})")
            print(f"  Remaining: {', '.join(experiment_names)}")
            for name in experiment_names:
                if name not in config["experiments"]:
                    print(f"Warning: experiment '{name}' no longer in config, skipping")
            experiment_names = [n for n in experiment_names if n in config["experiments"]]
        else:
            print("\nNo interrupted queue to resume. Starting fresh.\n")
            experiment_names = list(config["experiments"].keys())
    elif args and not all(a.startswith("--") for a in args):
        experiment_names = [a for a in args if not a.startswith("--")]
        for name in experiment_names:
            if name not in config["experiments"]:
                print(f"Unknown experiment: {name}")
                print(f"Available: {', '.join(config['experiments'].keys())}")
                return
    else:
        experiment_names = list(config["experiments"].keys())

    print(f"\nWill run {len(experiment_names)} experiment(s): {', '.join(experiment_names)}")
    print(f"Estimated time: ~{len(experiment_names) * 14} minutes (10 min train + ~4 min eval each)")
    print(f"Press Ctrl+C to gracefully stop after the current experiment.\n")

    consecutive_failures = 0

    with keep_awake():
        for idx, name in enumerate(experiment_names):
            save_progress(experiment_names[idx:], current_experiment=name)

            try:
                metrics = run_experiment(name, config)
            except KeyboardInterrupt:
                # Re-include the interrupted experiment so --resume retries it
                remaining = experiment_names[idx:]
                if remaining:
                    save_progress(remaining)
                    print(f"\n*** Queue stopped. {len(remaining)} experiment(s) remaining. ***")
                    print(f"*** Run with --resume to continue: {', '.join(remaining)} ***\n")
                else:
                    clear_progress()
                    print("\n*** That was the last experiment — queue complete despite interruption. ***\n")
                break

            if metrics:
                results.append(metrics)
                save_results(results)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    remaining = experiment_names[idx + 1:]
                    if remaining:
                        save_progress(remaining)
                    print(f"\n*** {consecutive_failures} consecutive failures — stopping queue. ***")
                    print(f"*** The same error is probably affecting all remaining experiments. ***")
                    print(f"*** Fix the issue, then run with --resume to continue. ***\n")
                    break
        else:
            clear_progress()

    print("\n\n")
    show_results()


if __name__ == "__main__":
    main()
