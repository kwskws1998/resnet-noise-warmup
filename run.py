"""Run a fixed, two-seed ResNet scale-up with epoch resume and deferred test evaluation."""

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import functional as F

import core

ROOT = Path(__file__).resolve().parent


def atomic_save(path, value):
    """Replace a checkpoint only after its complete serialization succeeds."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def synchronize(device):
    """Measure GPU execution rather than asynchronous kernel submission."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup(device_name):
    """Fail on unavailable requested hardware; keep precision common across conditions."""
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Full runs require a CUDA-enabled PyTorch environment.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def load_split(directory, split, manifest, device):
    """Read just one split; training never opens the test NPZ."""
    path = directory / f"{split}.npz"
    assert core.file_hash(path) == manifest["hashes"][path.name]
    with np.load(path) as arrays:
        inputs = torch.from_numpy(arrays["x"].transpose(0, 3, 1, 2).copy()).to(device, dtype=torch.float32) / 255.0
        labels = torch.from_numpy(arrays["y"].copy()).to(device)
    mean = inputs.new_tensor(manifest["mean"]).view(1, 3, 1, 1)
    std = inputs.new_tensor(manifest["std"]).view(1, 3, 1, 1)
    return (inputs - mean) / std, labels


def signature(config, manifest):
    """Bind checkpoints to operators, configuration, data, and software versions."""
    return {"sources": {name: core.file_hash(ROOT / name) for name in ["core.py", "run.py", "prepare.py", "protocol.json", "prior_test_files.json"]},
            "data": manifest["hashes"], "config": config, "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__}


def train_epoch(model, optimizer, inputs, labels, seed, task, epoch, config):
    """Pair sample order across conditions and synchronize only at epoch end."""
    model.train()
    generator = torch.Generator().manual_seed(1000000 + seed * 10000 + task * 100 + epoch)
    order = torch.randperm(len(inputs), generator=generator).to(inputs.device)
    total_loss = inputs.new_zeros(())
    total_correct = inputs.new_zeros(())
    for selected in order.split(config["batch_size"]):
        logits = model(inputs[selected])
        loss = F.cross_entropy(logits, labels[selected])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach() * len(selected)
        total_correct += logits.detach().argmax(1).eq(labels[selected]).sum()
    values = {"training_loss": float(total_loss.cpu()) / len(inputs), "training_accuracy": float(total_correct.cpu()) / len(inputs)}
    assert math.isfinite(values["training_loss"])
    return values


def summarize(seed, name, matrix, endpoints, curves, epochs):
    """Report acquisition and retention together, retaining paired seed identities."""
    forgetting = [max(matrix[stage][task] for stage in range(task, len(matrix))) - matrix[-1][task] for task in range(len(matrix) - 1)]
    result = {"seed": seed, "condition": name, "first_accuracy": endpoints[0][0]["accuracy"], "first_ece": endpoints[0][0]["ece"], "first_nll": endpoints[0][0]["nll"],
              "final_mean_accuracy": float(np.mean(matrix[-1])), "old_final_accuracy": float(np.mean(matrix[-1][:-1])), "mean_forgetting": float(np.mean(forgetting)),
              "later_acquired_accuracy": float(np.mean([matrix[i][i] for i in range(1, len(matrix))]))}
    if curves:
        result["later_auc"] = float(np.mean([np.trapezoid([row["accuracy"] for row in curves if row["task"] == task]) / epochs for task in range(1, len(matrix))]))
    return result


def run_condition(seed, condition, config, train, validation, device, output, run_signature, stop_after=None):
    """Resume exactly at an epoch boundary, with fresh Adam at every domain boundary."""
    name = condition["name"]
    directory = output / f"seed_{seed:02d}_{name}"
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "validation.json"
    if completed.exists():
        assert json.loads(completed.read_text())["signature"] == run_signature
        assert all((directory / f"task_{task}.pt").exists() for task in range(len(config["task_rotations_degrees"])))
        return
    model = core.make_model(seed, device)
    latest = directory / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        assert state["signature"] == run_signature
        model.load_state_dict(state["model"])
    else:
        initial_hash = core.parameter_hash(model)
        warm_config = dict(config, batch_size=config["warmup_batch_size"])
        warm = core.warmup(model, condition, warm_config, seed, device)
        generator = torch.Generator().manual_seed(config["split_seed"] + 37)
        order = torch.randperm(len(train[0]), generator=generator).to(device)
        calibration_config = dict(config, batch_size=config["batchnorm_calibration_batch_size"])
        limit = config["batchnorm_calibration_batches"] * config["batchnorm_calibration_batch_size"]
        core.calibrate_bn(model, train[0][order[:limit]], calibration_config)
        state = {"signature": run_signature, "next_task": 0, "next_epoch": 0, "optimizer": None, "model": model.state_dict(), "initial_parameter_sha256": initial_hash,
                 "warmup": warm, "curves": [], "matrix": [], "endpoints": []}
        atomic_save(latest, state)
    processed = 0
    for task in range(state["next_task"], len(config["task_rotations_degrees"])):
        quarter_turns = config["task_rotations_degrees"][task] // 90
        inputs = torch.rot90(train[0], quarter_turns, dims=(2, 3)).contiguous()
        validation_inputs = torch.rot90(validation[0], quarter_turns, dims=(2, 3)).contiguous()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        start_epoch = state["next_epoch"] if task == state["next_task"] else 0
        if start_epoch > 0:
            optimizer.load_state_dict(state["optimizer"])
        for epoch in range(start_epoch, config["epochs_per_task"] + 1):
            synchronize(device)
            start = time.perf_counter()
            values = train_epoch(model, optimizer, inputs, train[1], seed, task, epoch, config) if epoch else {}
            evaluation, _ = core.evaluate(model, validation_inputs, validation[1], config["eval_batch_size"])
            synchronize(device)
            record = {"task": task, "epoch": epoch, **values, **evaluation, "seconds": time.perf_counter() - start}
            state["curves"].append(record)
            state["next_task"], state["next_epoch"] = task, epoch + 1
            if epoch == config["epochs_per_task"]:
                endpoints = []
                for previous in range(task + 1):
                    domain = torch.rot90(validation[0], config["task_rotations_degrees"][previous] // 90, dims=(2, 3)).contiguous()
                    metrics, _ = core.evaluate(model, domain, validation[1], config["eval_batch_size"])
                    endpoints.append(metrics)
                state["endpoints"].append(endpoints)
                state["matrix"].append([item["accuracy"] for item in endpoints])
                atomic_save(directory / f"task_{task}.pt", {key: value.detach().cpu() for key, value in model.state_dict().items()})
                state["next_task"], state["next_epoch"] = task + 1, 0
            state["model"], state["optimizer"] = model.state_dict(), optimizer.state_dict()
            atomic_save(latest, state)
            print(json.dumps({"event": "validation", "seed": seed, "condition": name, **record}), flush=True)
            processed += 1
            if stop_after is not None and processed >= stop_after:
                return
    result = {key: value for key, value in state.items() if key not in {"model", "optimizer"}}
    result["summary"] = summarize(seed, name, state["matrix"], state["endpoints"], state["curves"], config["epochs_per_task"])
    core.save_json(completed, result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def write_summary(output, config, filename):
    """Keep a compact CSV with completed runs without imputing missing conditions."""
    rows = []
    for seed in config["seeds"]:
        for condition in config["conditions"]:
            path = output / f"seed_{seed:02d}_{condition['name']}" / filename
            if path.exists():
                rows.append(json.loads(path.read_text())["summary"])
    if rows:
        with (output / filename.replace(".json", "_summary.csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def finalize(config, manifest, device, output, run_signature):
    """Open official test only after every preregistered condition and seed completes."""
    grid = [(seed, condition["name"]) for seed in config["seeds"] for condition in config["conditions"]]
    for seed, name in grid:
        directory = output / f"seed_{seed:02d}_{name}"
        path = directory / "validation.json"
        if not path.exists():
            raise RuntimeError("Complete all six runs before test evaluation")
        assert json.loads(path.read_text())["signature"] == run_signature
        assert all((directory / f"task_{task}.pt").exists() for task in range(len(config["task_rotations_degrees"])))
    if not (output / "test_accessed.json").exists():
        core.save_json(output / "test_accessed.json", {"unix": time.time(), "signature": run_signature, "prior_poc_test_examples": 1000})
    inputs, labels = load_split(ROOT / "data", "test", manifest, device)
    with np.load(ROOT / "data/test.npz") as arrays:
        fresh = torch.from_numpy(arrays["fresh_mask"].copy())
    for seed, name in grid:
        directory = output / f"seed_{seed:02d}_{name}"
        model = core.make_model(seed, device)
        endpoints, fresh_endpoints, archive = [], [], {"labels": labels.cpu().numpy(), "fresh_mask": fresh.numpy()}
        for stage in range(len(config["task_rotations_degrees"])):
            model.load_state_dict(torch.load(directory / f"task_{stage}.pt", map_location=device, weights_only=True))
            row, fresh_row = [], []
            for task in range(stage + 1):
                domain = torch.rot90(inputs, config["task_rotations_degrees"][task] // 90, dims=(2, 3)).contiguous()
                values, logits = core.evaluate(model, domain, labels, config["eval_batch_size"])
                row.append(values)
                fresh_row.append(core.metrics(logits[fresh], labels.cpu()[fresh]))
                archive[f"after_{stage}_task_{task}"] = logits.numpy()
            endpoints.append(row)
            fresh_endpoints.append(fresh_row)
        for filename, values in [("test.json", endpoints), ("fresh_test.json", fresh_endpoints)]:
            matrix = [[item["accuracy"] for item in row] for row in values]
            summary = summarize(seed, name, matrix, values, [], config["epochs_per_task"])
            core.save_json(directory / filename, {"summary": summary, "endpoints": values, "matrix": matrix, "signature": run_signature})
        np.savez_compressed(directory / "test_logits.npz", **archive)
    write_summary(output, config, "test.json")
    write_summary(output, config, "fresh_test.json")


def benchmark(config, device, output):
    """Measure this host's real model before committing to all six runs; synthetic data only."""
    model = core.make_model(1, device)
    inputs = torch.randn(config["batch_size"], 3, 32, 32, device=device)
    labels = torch.arange(config["batch_size"], device=device) % 10
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    times = []
    for step in range(25):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), labels)
        loss.backward()
        optimizer.step()
        synchronize(device)
        if step >= 5:
            times.append(time.perf_counter() - start)
    updates = math.ceil(45000 / config["batch_size"]) * config["epochs_per_task"] * len(config["task_rotations_degrees"])
    hours = updates * len(config["seeds"]) * len(config["conditions"]) * float(np.median(times)) / 3600
    result = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "median_training_step_seconds": float(np.median(times)),
              "training_updates_per_run": updates, "six_run_training_only_hours": hours,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
              "excludes": "Validation, data movement/indexing, checkpoint IO, warm-up, downloads, provisioning and idle time. Benchmark peak excludes resident full dataset."}
    core.save_json(output / "benchmark.json", result)
    print(json.dumps(result), flush=True)


def main():
    """Expose benchmark, train, and a distinct final test-evaluation action."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "benchmark", "train", "finalize"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, choices=[1, 2])
    parser.add_argument("--conditions", nargs="+", choices=["none", "full", "full_var"])
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    device = setup(args.device)
    config = json.loads((ROOT / "protocol.json").read_text())
    assert config["seeds"] == [1, 2]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        print(json.dumps(core.smoke(config, device)))
        return
    if args.command == "benchmark":
        benchmark(config, device, args.output)
        return
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    assert manifest["protocol_sha256"] == core.file_hash(ROOT / "protocol.json")
    run_signature = signature(config, manifest)
    provenance_path = args.output / "provenance.json"
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text())["signature"] == run_signature
    else:
        core.save_json(provenance_path, {"signature": run_signature, "unix": time.time(), "python": sys.version, "platform": platform.platform(), "device": str(device),
                                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "reproducibility": "Paired initialization, noise and minibatches; no cross-device bitwise guarantee."})
    if args.command == "finalize":
        finalize(config, manifest, device, args.output, run_signature)
        return
    if (args.output / "test_accessed.json").exists():
        raise RuntimeError("Test already accessed for this experiment; no training after finalization")
    train = load_split(ROOT / "data", "train", manifest, device)
    validation = load_split(ROOT / "data", "validation", manifest, device)
    for seed in args.seeds or config["seeds"]:
        for condition in config["conditions"]:
            if args.conditions and condition["name"] not in args.conditions:
                continue
            run_condition(seed, condition, config, train, validation, device, args.output, run_signature)
            write_summary(args.output, config, "validation.json")


if __name__ == "__main__":
    main()
