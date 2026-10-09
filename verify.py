"""Exercise resume equivalence, split isolation, and final evaluation on tiny synthetic data."""

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import core
import run


def verify():
    """Test the real ResNet operators and training path without full CIFAR training."""
    device = run.setup("cpu")
    config = json.loads((run.ROOT / "protocol.json").read_text())
    config.update(epochs_per_task=1, batch_size=4, warmup_batch_size=4, eval_batch_size=4,
                  batchnorm_calibration_batches=1, batchnorm_calibration_batch_size=4,
                  task_rotations_degrees=[0, 90])
    for condition in config["conditions"]:
        condition["steps"] = 0 if condition["name"] == "none" else 2
    generator = torch.Generator().manual_seed(789)
    train = (torch.randn(8, 3, 32, 32, generator=generator), torch.arange(8) % 10)
    validation = (torch.randn(8, 3, 32, 32, generator=generator), torch.arange(8) % 10)
    signature = {"purpose": "synthetic integration verification", "config": copy.deepcopy(config)}
    with tempfile.TemporaryDirectory(prefix="resnet_scaleup_verify_") as temporary:
        root = Path(temporary)
        complete, resumed = root / "complete", root / "resumed"
        condition = config["conditions"][-1]
        run.run_condition(1, condition, config, train, validation, device, complete, signature)
        run.run_condition(1, condition, config, train, validation, device, resumed, signature, stop_after=2)
        run.run_condition(1, condition, config, train, validation, device, resumed, signature)
        left = torch.load(complete / "seed_01_full_var/task_1.pt", weights_only=True)
        right = torch.load(resumed / "seed_01_full_var/task_1.pt", weights_only=True)
        assert all(torch.equal(left[key], right[key]) for key in left), "Resume changed weights or buffers"
        a = json.loads((complete / "seed_01_full_var/validation.json").read_text())
        b = json.loads((resumed / "seed_01_full_var/validation.json").read_text())
        assert a["summary"] == b["summary"]
        assert len(b["curves"]) == 4, "Resume duplicated or dropped evaluation records"
        try:
            run.finalize(config, {}, device, complete, signature)
        except RuntimeError as error:
            assert "Complete all six" in str(error)
        else:
            raise AssertionError("Finalization opened test before the full grid completed")
        assert not (complete / "test_accessed.json").exists()
        for seed in config["seeds"]:
            for item in config["conditions"]:
                run.run_condition(seed, item, config, train, validation, device, complete, signature)
        initial_hashes = {}
        for seed in config["seeds"]:
            initial_hashes[seed] = {json.loads((complete / f"seed_{seed:02d}_{item['name']}" / "validation.json").read_text())["initial_parameter_sha256"] for item in config["conditions"]}
        assert all(len(values) == 1 for values in initial_hashes.values())
        run.write_summary(complete, config, "validation.json")
        assert len((complete / "validation_summary.csv").read_text().splitlines()) == 7
        original_root = run.ROOT
        try:
            run.ROOT = root
            (root / "data").mkdir()
            rng = np.random.default_rng(4)
            np.savez(root / "data/test.npz", x=rng.integers(0, 256, (8, 32, 32, 3), dtype=np.uint8),
                     y=np.arange(8, dtype=np.int64), fresh_mask=np.array([False, True, True, True, True, True, True, True]))
            manifest = {"hashes": {"test.npz": core.file_hash(root / "data/test.npz")}, "mean": [0.5] * 3, "std": [0.2] * 3}
            run.finalize(config, manifest, device, complete, signature)
            assert (complete / "test_accessed.json").exists()
            assert len((complete / "fresh_test_summary.csv").read_text().splitlines()) == 7
            data = np.load(complete / "seed_01_full_var/test_logits.npz")
            result = json.loads((complete / "seed_01_full_var/fresh_test.json").read_text())
            fresh = data["fresh_mask"]
            measured = core.metrics(torch.from_numpy(data["after_1_task_0"][fresh]), torch.from_numpy(data["labels"][fresh]))
            assert measured == result["endpoints"][1][0]
        finally:
            run.ROOT = original_root
    result = {"real_resnet_synthetic_integration": True, "exact_resume_parameters_and_bn": True,
              "resume_metrics": True, "finalize_requires_complete_grid": True, "paired_initialization": True,
              "six_run_summary": True, "final_test_evaluation": True, "fresh_subset_metrics_recomputed": True,
              "cuda_tested": False, "full_dataset_training_performed": False}
    core.save_json(Path(__file__).resolve().parent / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    verify()
