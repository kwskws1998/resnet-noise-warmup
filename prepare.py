"""Prepare full CIFAR-10 with separately stored, deferred-access test data."""

import argparse
import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from core import file_hash, save_json

ROOT = Path(__file__).resolve().parent
URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
SHA256 = "637c5814e11aefcb6ee76d5f59c67ddc8de7f5b5077502a195b0833d1e3e4441"


def prepare(archive=None):
    """Validate the source and fix all memberships before any scale-up training."""
    config = json.loads((ROOT / "protocol.json").read_text())
    output = ROOT / "data"
    output.mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["protocol_sha256"] == file_hash(ROOT / "protocol.json")
        assert all(file_hash(output / name) == digest for name, digest in manifest["hashes"].items())
        print("Existing prepared data validated.", flush=True)
        return
    archive = Path(archive) if archive else output / "cifar10_fastai.tgz"
    if not archive.exists():
        if archive != output / "cifar10_fastai.tgz":
            raise FileNotFoundError(archive)
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if file_hash(temporary) != SHA256:
            raise ValueError("Downloaded archive hash mismatch")
        temporary.replace(archive)
    assert file_hash(archive) == SHA256, "Archive hash mismatch"
    groups = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            parts = Path(member.name).parts
            if member.isfile() and len(parts) == 4 and parts[1] in {"train", "test"} and parts[-1].endswith(".png"):
                groups.setdefault((parts[1], parts[2]), []).append(member.name)
    classes = sorted({name for split, name in groups if split == "train"})
    assert len(classes) == 10
    rng = np.random.default_rng(config["split_seed"])
    names = {split: [] for split in ["train", "validation", "test"]}
    labels = {split: [] for split in names}
    locations = {}
    for label, name in enumerate(classes):
        training = np.array(sorted(groups[("train", name)]))
        testing = sorted(groups[("test", name)])
        assert len(training) == 5000 and len(testing) == 1000
        training = training[rng.permutation(len(training))]
        for split, files in [("train", training[:4500]), ("validation", training[4500:]), ("test", testing)]:
            for item in files:
                locations[str(item)] = (split, len(names[split]))
                names[split].append(str(item))
                labels[split].append(label)
    pixels = {split: np.empty((len(files), 32, 32, 3), dtype=np.uint8) for split, files in names.items()}
    seen = set()
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if member.name in locations:
                split, index = locations[member.name]
                with Image.open(io.BytesIO(handle.extractfile(member).read())) as image:
                    pixels[split][index] = np.asarray(image.convert("RGB"))
                seen.add(member.name)
    assert len(seen) == 60000
    assert not set(names["train"]) & set(names["validation"])
    prior = set(json.loads((ROOT / "prior_test_files.json").read_text()))
    assert len(prior) == 1000 and prior <= set(names["test"])
    fresh = np.array([name not in prior for name in names["test"]], dtype=bool)
    for split in names:
        extra = {"fresh_mask": fresh} if split == "test" else {}
        np.savez(output / f"{split}.npz", x=pixels[split], y=np.array(labels[split], dtype=np.int64), **extra)
    normalized = pixels["train"].astype(np.float32) / 255.0
    manifest = {"source_url": URL, "archive_sha256": SHA256, "protocol_sha256": file_hash(ROOT / "protocol.json"), "classes": classes,
                "counts": {split: len(files) for split, files in names.items()}, "files": names,
                "fresh_test_count": int(fresh.sum()), "prior_poc_test_count": int((~fresh).sum()),
                "mean": normalized.mean((0, 1, 2)).tolist(), "std": normalized.std((0, 1, 2)).tolist(),
                "hashes": {f"{split}.npz": file_hash(output / f"{split}.npz") for split in names}}
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"prepared": manifest["counts"], "fresh_test_count": manifest["fresh_test_count"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    prepare(parser.parse_args().archive)
