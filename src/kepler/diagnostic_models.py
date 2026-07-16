from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .config import ModelFamilyConfig
from .model import KeplerTransformer


class ParentExposedTransformer(KeplerTransformer):
    """The frozen Transformer body on [x1, x2, learned fixed query]."""

    def __init__(self, width: int, cfg: ModelFamilyConfig, vocab_size: int = 257) -> None:
        super().__init__(width, cfg, vocab_size)
        self.query_embedding = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)

    def forward(self, parents: Tensor, return_residuals: bool = False):
        if parents.ndim != 2 or parents.shape[1] != 2:
            raise ValueError("parents must have shape [batch, 2]")
        positions = torch.arange(3, device=parents.device)
        parent_embeddings = self.token_embedding(parents)
        query = self.query_embedding.expand(parents.shape[0], 1, self.width)
        x = torch.cat((parent_embeddings, query), dim=1)
        x = x + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)
        residuals = [x[:, -1]] if return_residuals else []
        for block in self.blocks:
            x = block(x)
            if return_residuals:
                residuals.append(x[:, -1])
        logits = self.lm_head(self.final_norm(x[:, -1]))
        if return_residuals:
            return logits, residuals
        return logits

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def core_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith(
                ("token_embedding", "position_embedding", "task_emb", "query_embedding")
            )
        )


class ConcatenatedParentMLP(nn.Module):
    """Separate parent embeddings followed by a fixed-depth composition MLP."""

    def __init__(
        self,
        embedding_width: int = 168,
        hidden_width: int = 732,
        depth: int = 4,
        vocab_size: int = 257,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.embedding_width = embedding_width
        self.hidden_width = hidden_width
        self.depth = depth
        self.vocab_size = vocab_size
        self.parent1_embedding = nn.Embedding(vocab_size, embedding_width)
        self.parent2_embedding = nn.Embedding(vocab_size, embedding_width)
        dimensions = [2 * embedding_width] + [hidden_width] * depth
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(depth)
        )
        self.activation = nn.GELU(approximate="tanh")
        self.final_norm = nn.LayerNorm(hidden_width)
        self.output_head = nn.Linear(hidden_width, vocab_size, bias=False)
        self.apply(KeplerTransformer._init_weights)

    def forward(self, parents: Tensor) -> Tensor:
        if parents.ndim != 2 or parents.shape[1] != 2:
            raise ValueError("parents must have shape [batch, 2]")
        x = torch.cat(
            (
                self.parent1_embedding(parents[:, 0]),
                self.parent2_embedding(parents[:, 1]),
            ),
            dim=-1,
        )
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_head(self.final_norm(x))

    def core_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith(("parent1_embedding", "parent2_embedding"))
        )

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "concatenated_parent_mlp",
            "embedding_width": self.embedding_width,
            "hidden_width": self.hidden_width,
            "fixed_depth": self.depth,
            "core_parameters": self.core_parameter_count(),
            "total_parameters": self.total_parameter_count(),
            "separate_parent_embeddings": True,
        }
