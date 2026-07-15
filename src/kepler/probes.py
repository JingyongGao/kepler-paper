from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DEFAULT_CONFIG, ExperimentConfig
from .data import OnlineEpisodeStream
from .topology import FiniteFieldRule


def fourier_targets(z: Tensor, p: int = 257) -> Tensor:
    """The fixed, complete real Fourier basis Φ(z), excluding DC."""
    if p % 2 == 0:
        raise ValueError("the registered basis assumes odd p")
    frequencies = torch.arange(1, (p - 1) // 2 + 1, device=z.device, dtype=torch.float32)
    angles = 2.0 * math.pi * z.float().unsqueeze(-1) * frequencies / p
    # Interleave [cos(mz), sin(mz)] for m=1..(p-1)/2 exactly as registered.
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(-2)


class LinearStateProbe(nn.Module):
    def __init__(self, width: int, p: int = 257) -> None:
        super().__init__()
        self.readout = nn.Linear(width, p)

    def forward(self, residual: Tensor) -> Tensor:
        return self.readout(residual)


class FourierStateProbe(nn.Module):
    """Linear readout to fixed Fourier coordinates; frequencies are not learned."""

    def __init__(self, width: int, p: int = 257) -> None:
        super().__init__()
        self.p = p
        self.readout = nn.Linear(width, p - 1)

    def forward(self, residual: Tensor) -> Tensor:
        return self.readout(residual)


class MLPStateProbe(nn.Module):
    def __init__(self, width: int, p: int = 257, hidden: int | None = None) -> None:
        super().__init__()
        hidden_width = hidden or min(512, 2 * width)
        self.net = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, p),
        )

    def forward(self, residual: Tensor) -> Tensor:
        return self.net(residual)


@dataclass(frozen=True)
class ProbeDataset:
    residuals: Tensor
    centers: Tensor


@torch.no_grad()
def extract_probe_dataset(
    model,
    rule: FiniteFieldRule,
    seed: int,
    examples: int,
    layer: int,
    device: str,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    batch_size: int = 32,
) -> ProbeDataset:
    """Extract a disjoint stream; callers must use distinct split seeds."""
    if not 0 <= layer <= cfg.model.n_layers:
        raise ValueError("layer indexes embedding output as 0 and block outputs as 1..L")
    scored = cfg.model.context_length - cfg.model.visible_prefix_length
    stream = OnlineEpisodeStream(
        rule,
        seed,
        cfg.model.visible_prefix_length,
        scored,
        cfg.field.d_max,
    )
    representations: list[Tensor] = []
    labels: list[Tensor] = []
    model.eval()
    seen = 0
    while seen < examples:
        count = min(batch_size, math.ceil((examples - seen) / scored))
        batch = stream.batch(count)
        inputs, _, centers, mask = batch.to_torch(device)
        _, residuals = model(inputs, return_residuals=True)
        r = residuals[layer][mask]
        z = centers[mask]
        keep = min(examples - seen, r.shape[0])
        representations.append(r[:keep].detach().float().cpu())
        labels.append(z[:keep].detach().cpu())
        seen += keep
    return ProbeDataset(torch.cat(representations), torch.cat(labels))


def _batches(dataset: ProbeDataset, batch_size: int, generator: torch.Generator):
    order = torch.randperm(dataset.centers.shape[0], generator=generator)
    for start in range(0, order.numel(), batch_size):
        idx = order[start : start + batch_size]
        yield dataset.residuals[idx], dataset.centers[idx]


def _classification_metrics(probe: nn.Module, data: ProbeDataset) -> dict[str, float]:
    probe.eval()
    with torch.no_grad():
        logits = probe(data.residuals)
        return {
            "cross_entropy_nats": float(F.cross_entropy(logits, data.centers)),
            "accuracy": float((logits.argmax(dim=-1) == data.centers).float().mean()),
        }


def train_classification_probe(
    probe: nn.Module,
    train: ProbeDataset,
    validation: ProbeDataset,
    test: ProbeDataset,
    epochs: int = 50,
    learning_rate: float = 1.0e-3,
    batch_size: int = 512,
    patience: int = 7,
    seed: int = 0,
) -> dict[str, object]:
    """Tune on validation and touch the disjoint test set once at the end."""
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    stale = 0
    epochs_run = 0
    for epoch in range(epochs):
        probe.train()
        for x, z in _batches(train, batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(x), z)
            loss.backward()
            optimizer.step()
        val = _classification_metrics(probe, validation)
        epochs_run = epoch + 1
        if val["cross_entropy_nats"] < best_loss:
            best_loss = val["cross_entropy_nats"]
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise AssertionError("probe training produced no state")
    probe.load_state_dict(best_state)
    return {
        "epochs": epochs_run,
        "validation": _classification_metrics(probe, validation),
        "test": _classification_metrics(probe, test),
    }


def _fourier_metrics(probe: FourierStateProbe, data: ProbeDataset) -> dict[str, float]:
    probe.eval()
    with torch.no_grad():
        predicted = probe(data.residuals)
        target = fourier_targets(data.centers, probe.p)
        mse = F.mse_loss(predicted, target)
        variance = torch.mean((target - target.mean(dim=0, keepdim=True)) ** 2)
        return {"mse": float(mse), "variance_normalized_r2": float(1.0 - mse / variance)}


def train_fourier_probe(
    probe: FourierStateProbe,
    train: ProbeDataset,
    validation: ProbeDataset,
    test: ProbeDataset,
    epochs: int = 50,
    learning_rate: float = 1.0e-3,
    batch_size: int = 512,
    patience: int = 7,
    seed: int = 0,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best = math.inf
    best_state: dict[str, Tensor] | None = None
    stale = 0
    epochs_run = 0
    for epoch in range(epochs):
        probe.train()
        for x, z in _batches(train, batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(probe(x), fourier_targets(z, probe.p))
            loss.backward()
            optimizer.step()
        score = _fourier_metrics(probe, validation)["mse"]
        epochs_run = epoch + 1
        if score < best:
            best = score
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise AssertionError("probe training produced no state")
    probe.load_state_dict(best_state)
    return {
        "epochs": epochs_run,
        "basis": "fixed real Fourier m=1..128 (DC excluded)",
        "validation": _fourier_metrics(probe, validation),
        "test": _fourier_metrics(probe, test),
    }
