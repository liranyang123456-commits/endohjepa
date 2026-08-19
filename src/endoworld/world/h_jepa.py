"""Endo-HJEPA: hierarchical joint-embedding world model for endoscopic video.

L1  short-horizon dense or pooled latent prediction (tissue, tool, camera)
L2  coarser mid-horizon prediction (anatomy / procedure timescale)
L3  action-conditioned predictor + energy head for latent MPC

v2 method improvements over the initial stack:
- factorised spatio-temporal L1 predictor (space+time attention over *all* tokens)
  instead of independent per-site prediction on a 16-token subset;
- uncertainty-weighted multi-task objective (learnable log-variances) instead of
  hand-tuned 0.5 / 0.1 / 0.05 coefficients;
- VICReg-style variance/covariance anti-collapse regulariser on dense predictions;
- the pooled L1 head is now trained in dense mode (previously its loss was dropped,
  so the pooled predictor used at eval time was effectively untrained).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from endoworld.world.energy import EnergyHead, contrastive_energy_loss
from endoworld.world.latent_action import LatentActionTokenizer
from endoworld.world.plan_mpc import latent_mpc


@dataclass
class HJEPAConfig:
    latent_dim: int = 384
    hidden_dim: int = 512
    n_heads: int = 8
    n_layers: int = 4
    n_domains: int = 4
    n_actions: int = 16
    history: int = 4
    horizon: int = 4
    l2_stride: int = 2
    dropout: float = 0.1
    spatial_keep: int = 256  # v2: use all spatial tokens by default
    ablation: str = "full"  # l1 | l1l2 | full
    predictor: str = "spacetime"  # spacetime | persite (legacy ablation)
    vicreg: float = 0.1  # weight of the anti-collapse regulariser
    unc_weight: bool = True  # uncertainty-weighted multi-task loss
    residual: bool = (
        True  # predict delta from the last token (persistence + correction)
    )
    l1_causal: bool = (
        True  # autoregressive causal L1 (GPT-style); beats query-token & GRU
    )
    query_mask: str = "block_causal"  # block_causal | parallel (legacy encoder-bypass)


def l2_future_targets(z_future: torch.Tensor, stride: int) -> torch.Tensor:
    """Select the end of each coarse future block (e.g. steps 2 and 4)."""
    if stride < 1:
        raise ValueError("stride must be positive")
    return z_future[:, stride - 1 :: stride]


class DomainConditioned(nn.Module):
    def __init__(self, dim: int, n_domains: int):
        super().__init__()
        self.embed = nn.Embedding(n_domains, dim)

    def forward(self, x, domain_id):
        return x + self.embed(domain_id).unsqueeze(1)


class TransformerPredictor(nn.Module):
    """Predict future pooled latents from a history, optionally + discrete actions."""

    def __init__(
        self, cfg: HJEPAConfig, action_cond: bool = False, horizon: int | None = None
    ):
        super().__init__()
        self.cfg = cfg
        self.horizon = cfg.horizon if horizon is None else horizon
        self.action_cond = action_cond
        if cfg.hidden_dim % cfg.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.query = nn.Parameter(torch.randn(1, self.horizon, cfg.hidden_dim) * 0.02)
        self.head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.residual = getattr(cfg, "residual", True)
        if action_cond:
            self.action_embed = nn.Embedding(cfg.n_actions, cfg.hidden_dim)

    @staticmethod
    def _positions(length: int, dim: int, device, dtype) -> torch.Tensor:
        """Parameter-free sinusoidal positions (keeps old checkpoints loadable)."""
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / max(dim, 1))
        )
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * scale)
        if dim > 1:
            pe[:, 1::2] = torch.cos(pos * scale[: pe[:, 1::2].shape[1]])
        return pe.to(dtype=dtype).unsqueeze(0)

    @staticmethod
    def _block_causal_mask(history: int, horizon: int, device) -> torch.Tensor:
        """Observed history is bidirectional; future queries are autoregressive."""
        total = history + horizon
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        # Observed tokens must not read predicted future queries.
        mask[:history, history:] = True
        # Query k can read all history and queries <= k, never later queries.
        mask[history:, history:] = torch.triu(
            torch.ones(horizon, horizon, dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask

    def forward(self, z_hist, domain_tok=None, actions=None):
        b = z_hist.size(0)
        if self.cfg.query_mask == "parallel":
            # Reproduce legacy checkpoints whose query branch bypassed the
            # Transformer. This mode is retained for provenance only; new
            # contextual query predictors must request ``block_causal``.
            pred = self.head(self.query.expand(b, -1, -1))
            if self.residual:
                pred = pred + z_hist[:, -1:].expand(-1, pred.size(1), -1)
            return pred
        if self.cfg.query_mask != "block_causal":
            raise ValueError(
                f"unknown query_mask={self.cfg.query_mask!r}; "
                "expected 'block_causal' or 'parallel'"
            )
        h = self.in_proj(z_hist)
        if domain_tok is not None:
            h = h + domain_tok
        q = self.query.expand(b, -1, -1)
        if self.action_cond:
            if actions is None:
                raise ValueError("action-conditioned predictor requires actions")
            if actions.size(1) < self.horizon:
                raise ValueError(
                    f"need {self.horizon} actions, received {actions.size(1)}"
                )
            # Align action k with future query k. It remains subject to the
            # block-causal query mask, so a later action cannot affect an
            # earlier predicted state.
            q = q + self.action_embed(actions[:, : self.horizon])
        seq = torch.cat([h, q], dim=1)
        seq = seq + self._positions(seq.size(1), seq.size(2), seq.device, seq.dtype)
        seq = self.encoder(
            seq,
            mask=self._block_causal_mask(z_hist.size(1), self.horizon, seq.device),
        )
        pred = seq[:, -self.horizon :]
        pred = self.head(pred)
        if self.residual:
            # predict the delta from the last observed token: persistence + correction
            pred = pred + z_hist[:, -1:].expand(-1, pred.size(1), -1)
        return pred


class SpatioTemporalBlock(nn.Module):
    """One factorised block: temporal attention per site, then spatial attention per step."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(
            dim,
            n_heads,
            dim * 4,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.spatial = nn.TransformerEncoderLayer(
            dim,
            n_heads,
            dim * 4,
            dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

    def forward(self, x):  # (B, T, N, D)
        b, t, n, d = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(b * n, t, d)
        xt = self.temporal(xt)
        x = xt.reshape(b, n, t, d).permute(0, 2, 1, 3)
        xs = x.reshape(b * t, n, d)
        xs = self.spatial(xs)
        return xs.reshape(b, t, n, d)


class SpatioTemporalPredictor(nn.Module):
    """Dense future-token prediction with factorised space-time attention.

    Unlike the legacy per-site predictor, spatial attention lets the model capture
    global camera motion (the dominant predictable signal in endoscopy) and uses
    every spatial token rather than a small strided subset.
    """

    def __init__(self, cfg: HJEPAConfig, max_t: int = 64, max_n: int = 1024):
        super().__init__()
        self.cfg = cfg
        self.horizon = cfg.horizon
        h = cfg.hidden_dim
        self.in_proj = nn.Linear(cfg.latent_dim, h)
        self.pos_t = nn.Parameter(torch.randn(1, max_t, 1, h) * 0.02)
        self.pos_n = nn.Parameter(torch.randn(1, 1, max_n, h) * 0.02)
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(h, cfg.n_heads, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.query = nn.Parameter(torch.randn(1, cfg.horizon, 1, h) * 0.02)
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, cfg.latent_dim)
        self.residual = getattr(cfg, "residual", True)

    def forward(self, z_hist, domain_vec):
        """z_hist (B, H, N, Dlatent), domain_vec (B, hidden) -> (B, horizon, N, Dlatent)."""
        b, t, n, _ = z_hist.shape
        x = self.in_proj(z_hist) + self.pos_t[:, :t] + self.pos_n[:, :, :n]
        if domain_vec is not None:
            x = x + domain_vec.view(b, 1, 1, -1)
        q = self.query.expand(b, -1, n, -1) + self.pos_n[:, :, :n]
        x = torch.cat([x, q], dim=1)
        for blk in self.blocks:
            x = blk(x)
        out = self.head(self.norm(x[:, -self.horizon :]))
        if self.residual:
            out = out + z_hist[:, -1:].expand(-1, out.size(1), -1, -1)
        return out


class CausalTemporalPredictor(nn.Module):
    """Autoregressive causal Transformer over time (GPT-style) for latent forecast.

    Unlike the query-token predictor (which predicts all future steps in parallel),
    this predicts the next token from the causal context and feeds it back, matching
    the recurrent inductive bias that makes GRUs strong on smooth latent sequences,
    while keeping attention capacity and domain conditioning.
    """

    def __init__(self, cfg: HJEPAConfig, max_t: int = 64):
        super().__init__()
        self.cfg = cfg
        self.horizon = cfg.horizon
        h = cfg.hidden_dim
        self.in_proj = nn.Linear(cfg.latent_dim, h)
        self.pos = nn.Parameter(torch.randn(1, max_t, h) * 0.02)
        layer = nn.TransformerEncoderLayer(
            h,
            cfg.n_heads,
            h * 4,
            cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, cfg.latent_dim)
        self.residual = getattr(cfg, "residual", True)

    @staticmethod
    def _causal_mask(t: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, z_hist, domain_tok=None, horizon: int | None = None):
        horizon = horizon or self.horizon
        seq = z_hist
        preds = []
        for _ in range(horizon):
            h = self.in_proj(seq) + self.pos[:, : seq.size(1)]
            if domain_tok is not None:
                h = h + domain_tok
            out = self.blocks(h, mask=self._causal_mask(h.size(1), h.device))
            nxt = self.head(self.norm(out[:, -1:]))
            if self.residual:
                nxt = nxt + seq[:, -1:]
            preds.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)
        return torch.cat(preds, dim=1)


def subsample_spatial(z: torch.Tensor, keep: int) -> torch.Tensor:
    """z (B,T,N,D) -> (B,T,keep,D) with a fixed strided subset of spatial tokens."""
    n = z.size(2)
    if n <= keep:
        return z
    idx = torch.linspace(0, n - 1, keep, device=z.device).round().long()
    return z.index_select(2, idx)


def vicreg_loss(pred: torch.Tensor) -> torch.Tensor:
    """Variance floor + covariance decorrelation on dense predictions (anti-collapse)."""
    x = pred.reshape(-1, pred.size(-1)).float()
    std = torch.sqrt(x.var(dim=0) + 1e-4)
    v = F.relu(1.0 - std).mean()
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / max(x.size(0) - 1, 1)
    d = cov.size(0)
    off = cov - torch.diag(torch.diag(cov))
    c = off.pow(2).sum() / d
    return v + c


class EndoHJEPA(nn.Module):
    def __init__(self, cfg: HJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.domain = DomainConditioned(cfg.hidden_dim, cfg.n_domains)
        self.l1 = (
            CausalTemporalPredictor(cfg)
            if getattr(cfg, "l1_causal", False)
            else TransformerPredictor(cfg, action_cond=False)
        )
        self.l1_dense = (
            SpatioTemporalPredictor(cfg) if cfg.predictor == "spacetime" else None
        )
        cfg_l2 = replace(cfg, horizon=max(cfg.horizon // cfg.l2_stride, 1))
        self.l2 = TransformerPredictor(cfg_l2, action_cond=False)
        self.l3 = TransformerPredictor(cfg, action_cond=True)
        self.actions = LatentActionTokenizer(cfg.latent_dim, cfg.n_actions)
        self.energy = EnergyHead(cfg.latent_dim, cfg.n_actions, cfg.hidden_dim)
        self.l2_cfg = cfg_l2
        # learnable log-variances for uncertainty-weighted multi-task learning
        self.log_vars = nn.ParameterDict(
            {
                k: nn.Parameter(torch.zeros(()))
                for k in ("l1", "l1_pool", "l2", "l3", "energy", "commit")
            }
        )

    def _dom(self, z_hist, domain_id):
        return self.domain.embed(domain_id).unsqueeze(1)

    def _uw(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        """Uncertainty weighting: exp(-s)*loss + s. Falls back to the raw loss."""
        if not getattr(self.cfg, "unc_weight", False) or name not in self.log_vars:
            return loss
        s = self.log_vars[name]
        return torch.exp(-s) * loss + s

    def forward_l1(self, z_hist, domain_id):
        return self.l1(z_hist, domain_tok=self._dom(z_hist, domain_id))

    def forward_l1_dense(self, z_hist, domain_id):
        """z_hist (B, history, N, D) -> pred (B, horizon, N, D)."""
        if self.l1_dense is not None:
            return self.l1_dense(z_hist, self.domain.embed(domain_id))
        b, h, n, d = z_hist.shape
        z_flat = z_hist.permute(0, 2, 1, 3).reshape(b * n, h, d)
        dom = domain_id.unsqueeze(1).expand(-1, n).reshape(b * n)
        pred = self.forward_l1(z_flat, dom)
        return pred.view(b, n, pred.size(1), d).permute(0, 2, 1, 3)

    def forward_l2(self, z_hist, domain_id):
        t = z_hist.size(1)
        s = self.cfg.l2_stride
        t2 = t - (t % s)
        if t2 < s:
            pooled = z_hist
        else:
            pooled = z_hist[:, :t2].reshape(z_hist.size(0), t2 // s, s, -1).mean(dim=2)
        return self.l2(pooled, domain_tok=self._dom(pooled, domain_id))

    def tokenize_actions(self, z_seq):
        idx, quant, commit = self.actions(
            z_seq[:, :-1].reshape(-1, z_seq.size(-1)),
            z_seq[:, 1:].reshape(-1, z_seq.size(-1)),
        )
        b, t, d = z_seq.shape
        idx = idx.view(b, t - 1)
        return idx, commit

    def forward_l3(self, z_hist, actions, domain_id):
        return self.l3(z_hist, domain_tok=self._dom(z_hist, domain_id), actions=actions)

    def energy_loss(self, z_t, action, z_pos, z_neg):
        return contrastive_energy_loss(self.energy, z_t, action, z_pos, z_neg)

    def losses(self, z, domain_id, history: int, horizon: int):
        """z (B,T,D) pooled. Returns dict of scalar losses for the configured ablation."""
        z_hist, z_future = z[:, :history], z[:, history : history + horizon]
        pred1 = self.forward_l1(z_hist, domain_id)
        h1 = min(pred1.size(1), z_future.size(1))
        out = {"l1": F.smooth_l1_loss(pred1[:, :h1], z_future[:, :h1]), "pred": pred1}
        mode = self.cfg.ablation
        if mode in ("l1l2", "full"):
            z_l2_tgt = l2_future_targets(z_future, self.cfg.l2_stride)
            pred2 = self.forward_l2(z_hist, domain_id)
            h2 = min(pred2.size(1), z_l2_tgt.size(1))
            out["l2"] = F.smooth_l1_loss(pred2[:, :h2], z_l2_tgt[:, :h2])
        if mode == "full":
            act_idx, commit = self.tokenize_actions(z[:, : history + horizon])
            pred3 = self.forward_l3(
                z_hist, act_idx[:, history - 1 : history - 1 + horizon], domain_id
            )
            out["l3"] = F.smooth_l1_loss(pred3, z_future)
            z_t = z_hist[:, -1]
            a = act_idx[:, history - 1]
            z_pos = z_future[:, 0]
            z_neg = z_pos.roll(1, dims=0)
            loss_e, _, _ = self.energy_loss(z_t, a, z_pos, z_neg)
            out["energy"] = loss_e
            out["commit"] = commit
        total = self._uw("l1", out["l1"])
        if "l2" in out:
            total = total + self._uw("l2", out["l2"])
        if "l3" in out:
            total = (
                total
                + self._uw("l3", out["l3"])
                + self._uw("energy", out["energy"])
                + self._uw("commit", out["commit"])
            )
        out["total"] = total
        return out

    def losses_dense(self, z, domain_id, history: int, horizon: int):
        """z (B,T,N,D). L1 on dense tokens; L2/L3/energy on spatially pooled tokens.

        The pooled L1 head is trained here too (its loss was previously dropped),
        which is what the pooled eval path (`forward_l1`) relies on.
        """
        if self.l1_dense is not None:
            z_hist, z_future = z[:, :history], z[:, history : history + horizon]
        else:
            z = subsample_spatial(z, self.cfg.spatial_keep)
            z_hist, z_future = z[:, :history], z[:, history : history + horizon]
        pred1 = self.forward_l1_dense(z_hist, domain_id)
        h1 = min(pred1.size(1), z_future.size(1))
        out = {
            "l1": F.smooth_l1_loss(pred1[:, :h1], z_future[:, :h1]),
            "pred": pred1.mean(dim=2),
        }
        if self.l1_dense is not None and self.cfg.vicreg > 0:
            out["vicreg"] = vicreg_loss(pred1)

        z_pool = z.mean(dim=2) if z.dim() == 4 else z
        pooled = self.losses(z_pool, domain_id, history, horizon)
        out["l1_pool"] = pooled["l1"]
        mode = self.cfg.ablation
        if mode in ("l1l2", "full"):
            out["l2"] = pooled["l2"]
        if mode == "full":
            out["l3"] = pooled["l3"]
            out["energy"] = pooled["energy"]
            out["commit"] = pooled["commit"]

        total = self._uw("l1", out["l1"]) + self._uw("l1_pool", out["l1_pool"])
        if "l2" in out:
            total = total + self._uw("l2", out["l2"])
        if "l3" in out:
            total = (
                total
                + self._uw("l3", out["l3"])
                + self._uw("energy", out["energy"])
                + self._uw("commit", out["commit"])
            )
        if "vicreg" in out:
            total = total + self.cfg.vicreg * out["vicreg"]
        out["total"] = total
        return out

    @torch.no_grad()
    def plan(self, z_hist, z_goal, domain_id, n_samples: int = 32, steps: int = 4):
        return latent_mpc(
            self, z_hist, z_goal, domain_id, n_samples=n_samples, steps=steps
        )


def persistence_baseline(z_hist, horizon: int):
    return z_hist[:, -1:].expand(-1, horizon, -1)
