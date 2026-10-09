"""Model and warm-up operators copied unchanged from the completed ResNet PoC."""

import hashlib
import json
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

def save_json(path, value):
    """Write JSON without silently allowing NaNs."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def file_hash(path):
    """Return a SHA-256 digest for a saved source or data artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parameter_hash(model, backbone_only=False):
    """Hash parameters only; BatchNorm running buffers are controlled separately."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if backbone_only and name.startswith("fc."):
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def make_model(seed, device):
    """Use full-width ResNet-18 residual blocks with a CIFAR-appropriate input stem."""
    torch.manual_seed(seed)
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model.to(device)

def forward_features(model, inputs):
    """Expose the original pooled 512-dimensional ResNet representation."""
    x = model.relu(model.bn1(model.conv1(inputs)))
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return torch.flatten(model.avgpool(x), 1)

def bn_state(model):
    """Copy only BatchNorm running moments and counters."""
    return {name: buffer.detach().clone() for name, buffer in model.named_buffers()}

def restore_bn(model, state):
    """Restore BatchNorm buffers without touching learned parameters."""
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(state[name])

def calibrate_bn(model, inputs, config):
    """Reset and cumulatively estimate BN moments from fixed unlabeled training inputs."""
    momenta = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            momenta[name] = layer.momentum
            layer.reset_running_stats()
            layer.momentum = None
    model.train()
    limit = config["batchnorm_calibration_batches"] * config["batch_size"]
    with torch.no_grad():
        for inputs_batch in inputs[:limit].split(config["batch_size"]):
            model(inputs_batch)
    for name, layer in model.named_modules():
        if name in momenta:
            layer.momentum = momenta[name]
    model.eval()

def metrics(logits, labels):
    """Calculate proper scores and ten equal-width-bin ECE on CPU."""
    probabilities = logits.softmax(1)
    confidence, predicted = probabilities.max(1)
    correct = predicted.eq(labels).float()
    bins = torch.clamp((10 * confidence).long(), max=9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if selected.any():
            ece += float(selected.float().mean() * (correct[selected].mean() - confidence[selected].mean()).abs())
    return {"accuracy": float(correct.mean()), "ece": ece, "nll": float(F.cross_entropy(logits, labels)), "confidence": float(confidence.mean()), "entropy": float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(1).mean())}

def evaluate(model, inputs, labels, batch_size):
    """Evaluate in inference mode without updating BatchNorm buffers."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(batch).cpu() for batch in inputs.split(batch_size)], 0)
    return metrics(logits, labels.cpu()), logits

def feature_diagnostics(model, inputs, batch_size):
    """Describe representations on a fixed real-input probe after common BN calibration."""
    model.eval()
    with torch.no_grad():
        features = torch.cat([forward_features(model, batch).cpu() for batch in inputs.split(batch_size)], 0)
    singular = torch.linalg.svdvals(features - features.mean(0, keepdim=True))
    probabilities = singular / singular.sum().clamp_min(1e-12)
    rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()) if float(singular.sum()) > 1e-12 else torch.tensor(0.0)
    return {"rms": float(features.square().mean().sqrt()), "std_mean": float(features.std(0, unbiased=False).mean()), "effective_rank": float(rank), "dead_fraction": float((features.abs().amax(0) < 1e-8).float().mean()), "head_norm": float(model.fc.weight.detach().norm().cpu())}

def warmup(model, condition, config, seed, device):
    """Train fresh random noise, optionally constraining pooled feature variance."""
    kind = condition["kind"]
    if kind == "none":
        return {"seconds": 0.0, "checkpoints": [], "head_only_parameters_unchanged": None}
    initial_backbone_hash = parameter_hash(model, backbone_only=True) if kind == "head" else None
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(kind != "head" or name.startswith("fc."))
    model.train()
    reference_generator = torch.Generator().manual_seed(93000 + seed)
    reference = torch.randn(config["batch_size"], 3, 32, 32, generator=reference_generator).to(device)
    saved_bn = bn_state(model)
    with torch.no_grad():
        reference_std = forward_features(model, reference).std(0, unbiased=False).detach()
    restore_bn(model, saved_bn)
    generator = torch.Generator().manual_seed(90000 + seed)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=config["warmup_learning_rate"], weight_decay=0)
    checkpoints = []
    start = time.perf_counter()
    for step in range(condition["steps"]):
        inputs = torch.randn(config["batch_size"], 3, 32, 32, generator=generator).to(device)
        labels = torch.randint(10, (config["batch_size"],), generator=generator).to(device)
        features = forward_features(model, inputs)
        logits = model.fc(features)
        cross_entropy = F.cross_entropy(logits, labels)
        penalty = features.new_zeros(())
        if kind == "full_var":
            floor = config["variance_floor_fraction"] * reference_std
            standard_deviation = (features.var(0, unbiased=False) + 1e-6).sqrt()
            penalty = F.relu(floor - standard_deviation).square().mean()
        loss = cross_entropy + config["variance_penalty_weight"] * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in {0, 49, 99, condition["steps"] - 1}:
            record = {"step": step + 1, "cross_entropy": float(cross_entropy.detach().cpu()), "variance_penalty": float(penalty.detach().cpu())}
            assert math.isfinite(record["cross_entropy"])
            checkpoints.append(record)
            print(json.dumps({"event": "warmup", "seed": seed, "condition": condition["name"], **record}), flush=True)
    unchanged = None
    if kind == "head":
        unchanged = initial_backbone_hash == parameter_hash(model, backbone_only=True)
        assert unchanged, "Classifier-only warm-up changed backbone parameters"
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    return {"seconds": time.perf_counter() - start, "checkpoints": checkpoints, "head_only_parameters_unchanged": unchanged}

def smoke(config, device):
    """Check feature extraction, frozen parameters, and normalization-buffer restoration."""
    model = make_model(1, device)
    generator = torch.Generator().manual_seed(32)
    inputs = torch.randn(8, 3, 32, 32, generator=generator).to(device)
    model.eval()
    with torch.no_grad():
        expected = model(inputs)
        actual = model.fc(forward_features(model, inputs))
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
    buffers = bn_state(model)
    momenta = {name: layer.momentum for name, layer in model.named_modules() if isinstance(layer, nn.BatchNorm2d)}
    local_config = dict(config, batchnorm_calibration_batches=1, batch_size=8)
    calibrate_bn(model, inputs, local_config)
    restore_bn(model, buffers)
    assert all(torch.equal(buffer, buffers[name]) for name, buffer in model.named_buffers())
    assert all(layer.momentum == momenta[name] for name, layer in model.named_modules() if name in momenta)
    result = warmup(model, {"name": "head_smoke", "kind": "head", "steps": 2}, local_config, 1, device)
    assert result["head_only_parameters_unchanged"]
    assert all(parameter.requires_grad for parameter in model.parameters())
    return {"forward_feature_equivalence": True, "bn_buffer_restoration": True, "bn_momentum_restoration": True, "head_only_parameter_freeze": True, "real_training_parameters_unfrozen": True}
