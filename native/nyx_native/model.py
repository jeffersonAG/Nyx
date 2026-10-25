"""Transformer causal propio (sin modelos preentrenados).

Las capas, embeddings, atención, normalización y pesos nacen en este módulo;
autograd y AdamW de PyTorch calculan y aplican gradientes reales.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tokenizer import VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    context: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError("Vocabulario incompatible con el tokenizador de bytes.")
        if not 16 <= self.context <= 1024:
            raise ValueError("El contexto debe estar entre 16 y 1024.")
        if not 32 <= self.d_model <= 512:
            raise ValueError("d_model fuera de rango.")
        if not 1 <= self.n_heads <= 16 or self.d_model % self.n_heads:
            raise ValueError("d_model debe ser divisible por n_heads.")
        if not 1 <= self.n_layers <= 16:
            raise ValueError("n_layers fuera de rango.")
        if not 0 <= self.dropout < 0.8:
            raise ValueError("dropout fuera de rango.")

    def as_dict(self) -> dict:
        return asdict(self)


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.n_heads
        self.width = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3, bias=False)
        self.project = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.residual_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, size = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq, self.heads, self.width).transpose(1, 2)
        k = k.view(batch, seq, self.heads, self.width).transpose(1, 2)
        v = v.view(batch, seq, self.heads, self.width).transpose(1, 2)
        # Una máscara causal impide observar tokens futuros.
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        joined = attended.transpose(1, 2).contiguous().view(batch, seq, size)
        return self.residual_dropout(self.project(joined))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(cfg.d_model)
        self.attention = CausalAttention(cfg)
        self.norm_mlp = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class NativeTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.positions = nn.Embedding(cfg.context, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Vinculación de pesos: embeddings de entrada y proyección de salida.
        self.output.weight = self.tokens.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, targets: Tensor | None = None
                ) -> tuple[Tensor, Tensor | None]:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("Se esperan IDs de tokens int64 con forma [batch, seq].")
        batch, seq = tokens.shape
        if not 1 <= seq <= self.config.context:
            raise ValueError("La secuencia supera el contexto del modelo.")
        if tokens.numel() and (tokens.min().item() < 0
                               or tokens.max().item() >= self.config.vocab_size):
            raise ValueError("Token fuera del vocabulario.")
        offsets = torch.arange(seq, device=tokens.device)
        x = self.dropout(self.tokens(tokens) + self.positions(offsets))
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != tokens.shape:
                raise ValueError("Targets deben coincidir en forma con tokens.")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=258)
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: Tensor, *, max_new_tokens: int = 80,
                 temperature: float = 0.9, top_k: int = 40,
                 seed: int | None = None) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError("La generación requiere un prompt [1, N].")
        if not 1 <= max_new_tokens <= 256:
            raise ValueError("max_new_tokens fuera de rango.")
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("temperature fuera de rango.")
        if not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k fuera de rango.")
        generator = torch.Generator(device=tokens.device)
        if seed is not None:
            generator.manual_seed(seed)
        self.eval()
        for _ in range(max_new_tokens):
            conditioned = tokens[:, -self.config.context:]
            logits, _ = self(conditioned)
            logits = logits[:, -1, :] / temperature
            logits[:, 258] = float("-inf")  # Nunca generar PAD.
            if top_k < logits.shape[-1]:
                threshold = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1, generator=generator)
            tokens = torch.cat((tokens, next_token), dim=1)
            if next_token.item() == 257:
                break
        return tokens

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
