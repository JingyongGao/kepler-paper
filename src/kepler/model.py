from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelFamilyConfig


def core_parameter_count_for_width(width: int, cfg: ModelFamilyConfig) -> int:
    """Exact Transformer-body count for the implementation below."""
    r = cfg.mlp_ratio
    if cfg.bias:
        per_block = (4 + 2 * r) * width * width + (r + 9) * width
    else:
        per_block = (4 + 2 * r) * width * width + 4 * width
    # Final LayerNorm remains affine regardless of linear-layer bias policy.
    return cfg.n_layers * per_block + 2 * width


def select_width(target_core_parameters: int, cfg: ModelFamilyConfig) -> int:
    """Choose the closest legal width while keeping head geometry fixed."""
    step = cfg.n_heads
    candidates = range(step, 4096 + step, step)
    return min(
        candidates,
        key=lambda w: abs(core_parameter_count_for_width(w, cfg) - target_core_parameters),
    )


@dataclass(frozen=True)
class ModelSize:
    target_core_parameters: int
    width: int
    actual_core_parameters: int
    total_parameters: int
    conditioner_parameters: int = 0


def size_spec(target: int, cfg: ModelFamilyConfig, vocab_size: int = 257) -> ModelSize:
    width = select_width(target, cfg)
    core = core_parameter_count_for_width(width, cfg)
    embeddings = (vocab_size + cfg.context_length) * width
    # LM head is tied and bias-free, hence adds no distinct parameters.
    conditioner = cfg.num_tasks * width
    total = core + embeddings + conditioner
    return ModelSize(target, width, core, total, conditioner)


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, n_heads: int, dropout: float, bias: bool) -> None:
        super().__init__()
        if width % n_heads:
            raise ValueError("width must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.out = nn.Linear(width, width, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.out(y.transpose(1, 2).contiguous().reshape(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, width: int, cfg: ModelFamilyConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, cfg.n_heads, cfg.dropout, cfg.bias)
        self.norm2 = nn.LayerNorm(width)
        hidden = cfg.mlp_ratio * width
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden, bias=cfg.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )
        self.residual_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.residual_dropout(self.attn(self.norm1(x)))
        return x + self.mlp(self.norm2(x))


class KeplerTransformer(nn.Module):
    def __init__(self, width: int, cfg: ModelFamilyConfig, vocab_size: int = 257) -> None:
        super().__init__()
        self.width = width
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(cfg.context_length, width)
        self.task_emb = nn.Embedding(cfg.num_tasks, width) if cfg.num_tasks else None
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(width, cfg) for _ in range(cfg.n_layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        if self.task_emb is not None:
            nn.init.zeros_(self.task_emb.weight)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: Tensor,
        return_residuals: bool = False,
        task_id: int | Tensor | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        _, length = tokens.shape
        if length > self.cfg.context_length:
            raise ValueError("sequence exceeds frozen context length")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        if task_id is not None:
            if self.task_emb is None:
                raise ValueError("task_id was provided but task conditioning is disabled")
            ids = torch.as_tensor(task_id, dtype=torch.long, device=tokens.device)
            if ids.ndim == 0:
                ids = ids.expand(tokens.shape[0])
            if ids.ndim != 1 or ids.shape[0] != tokens.shape[0]:
                raise ValueError("task_id must be a scalar or have shape [batch]")
            x = x + self.task_emb(ids)[:, None, :]
        x = self.embedding_dropout(x)
        residuals = [x] if return_residuals else []
        for block in self.blocks:
            x = block(x)
            if return_residuals:
                residuals.append(x)
        logits = self.lm_head(self.final_norm(x))
        if return_residuals:
            return logits, residuals
        return logits

    def core_parameter_count(self) -> int:
        names = (
            (name, p)
            for name, p in self.named_parameters()
            if not name.startswith("token_embedding")
            and not name.startswith("position_embedding")
            and not name.startswith("task_emb")
        )
        return sum(p.numel() for _, p in names)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_language_model_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("logits, targets, and mask are not aligned")
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    selected = per_token[mask]
    if selected.numel() == 0:
        raise ValueError("loss mask selects no tokens")
    return selected.mean()


def make_model(target: int, cfg: ModelFamilyConfig, vocab_size: int = 257) -> KeplerTransformer:
    spec = size_spec(target, cfg, vocab_size)
    model = KeplerTransformer(spec.width, cfg, vocab_size)
    if model.core_parameter_count() != spec.actual_core_parameters:
        raise AssertionError("analytical and instantiated core counts disagree")
    return model


def model_metadata(model: KeplerTransformer) -> dict[str, Any]:
    return {
        "width": model.width,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "context_length": model.cfg.context_length,
        "position_encoding": model.cfg.position_encoding,
        "core_parameters": model.core_parameter_count(),
        "total_parameters": model.total_parameter_count(),
        "tie_embeddings": model.cfg.tie_embeddings,
        "num_tasks": model.cfg.num_tasks,
        "conditioner_parameters": 0 if model.task_emb is None else model.task_emb.weight.numel(),
    }


def load_base_state_into_conditioned(
    conditioned_model: KeplerTransformer, base_state: dict[str, Tensor]
) -> None:
    """Migrate a Phase 2/3 state, allowing only task_emb.weight to be absent."""
    if conditioned_model.task_emb is None:
        raise ValueError("the destination model has no task conditioner")
    incompatible = conditioned_model.load_state_dict(base_state, strict=False)
    if set(incompatible.missing_keys) != {"task_emb.weight"} or incompatible.unexpected_keys:
        raise ValueError(
            "base checkpoint mismatch: only task_emb.weight may be missing; "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    nn.init.zeros_(conditioned_model.task_emb.weight)


@torch.no_grad()
def assert_conditioner_attachment_equivalence(
    base_model: KeplerTransformer,
    conditioned_model: KeplerTransformer,
    tokens: Tensor,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> dict[str, float]:
    """Preregistered FP32 check immediately after zero-row attachment."""
    base_model.eval().float()
    conditioned_model.eval().float()
    device = next(conditioned_model.parameters()).device
    tokens = tokens.to(device)
    base_logits = base_model(tokens.to(next(base_model.parameters()).device)).float().cpu()
    task_a = conditioned_model(tokens, task_id=0).float().cpu()
    task_b = conditioned_model(tokens, task_id=1).float().cpu()
    differences = {
        "base_vs_task_a_max_abs": float((base_logits - task_a).abs().max()),
        "base_vs_task_b_max_abs": float((base_logits - task_b).abs().max()),
        "task_a_vs_task_b_max_abs": float((task_a - task_b).abs().max()),
    }
    if not (
        torch.allclose(base_logits, task_a, atol=atol, rtol=rtol)
        and torch.allclose(base_logits, task_b, atol=atol, rtol=rtol)
    ):
        raise AssertionError(f"zero conditioner changed logits: {differences}")
    return differences
