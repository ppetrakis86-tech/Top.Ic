"""Lightstorm, Γύρος 1: μικρό transformer με μία υποδοχή MoE στο layer 3.

Κλειδωμένα (Notion Γ1·1, Mem Top.Ic/alternatives v8):
d_model 256 · 4 layers · 4 heads · context 256 · tied embeddings · GELU · FFN 1× στα 1, 2, 4.
Layer 3: φύλλο 8×8 σε τόρο, expert 256→64→256, k=5 (νικητής + σταυρός).

Σκέλη στην υποδοχή:
  A   softmax top-5 + load balancing (baseline, χωρίς τοπογραφία)
  B   SOM router: ευκλείδεια, κλειδιά με κανόνα SOM, experts με backprop, κανένα spatial loss
  C1  B + τοπικός κόρος (EMA θολωμένο στον σταυρό)
  C2  B + καθαρός DeSieno (EMA ανά expert)
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import metrics

ARMS = ("A", "B", "C1", "C2", "E")    # E: πείραμα 9β, SOM με βάρος το λάθος κάθε token


@dataclass
class Config:
    vocab: int = 14679
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ctx: int = 256
    ffn: int = 256            # FFN 1×
    moe_layer: int = 2        # 0-based: το 3ο layer
    side: int = 8
    k: int = 5
    expert_hidden: int = 64
    arm: str = "B"
    lb_coef: float = 0.01     # μόνο στο A
    ema_alpha: float = 1e-3   # C1/C2
    beta: float = 0.3         # C1/C2
    seed: int = 0             # αρχικά κλειδιά του SOM (seed 0 = τα κλειδιά του πρώτου τρεξίματος)
    key_init: str = "white"   # πείραμα 9: "smooth" = χαρτογραφικός θόρυβος (ομαλό τυχαίο έδαφος)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class FFN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x))), {}


class Experts(nn.Module):
    """64 μικρά FFN 256→64→256. Τρέχουν μόνο για τα (token, slot) που τους έλαχαν."""

    def __init__(self, n, d, hidden):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n, d, hidden) * d ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(n, hidden))
        self.w2 = nn.Parameter(torch.randn(n, hidden, d) * hidden ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(n, d))

    def forward(self, x, idx, weight):
        """x (N, d) · idx (N, k) experts · weight (N, k) -> (N, d)"""
        N, k = idx.shape
        out = torch.zeros_like(x)
        flat_idx, flat_w = idx.reshape(-1), weight.reshape(-1)
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        for e in torch.unique(flat_idx).tolist():
            sel = (flat_idx == e).nonzero(as_tuple=True)[0]
            t = tok[sel]
            h = F.gelu(x[t] @ self.w1[e] + self.b1[e]) @ self.w2[e] + self.b2[e]
            out.index_add_(0, t, h * flat_w[sel, None])
        return out


class SoftmaxMoE(nn.Module):
    """Σκέλος A: κλασικός router, softmax top-k, load balancing (Switch)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.n = cfg, cfg.side ** 2
        self.gate = nn.Linear(cfg.d_model, self.n, bias=False)
        self.experts = Experts(self.n, cfg.d_model, cfg.expert_hidden)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        probs = self.gate(x).softmax(-1)
        w, idx = probs.topk(self.cfg.k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        y = self.experts(x, idx, w)
        frac = F.one_hot(idx[:, 0], self.n).float().mean(0)
        lb = self.n * (frac * probs.mean(0)).sum()
        return y.reshape(shape), {"winner": idx[:, 0], "members": idx, "lb": lb}


def smooth_keys(side, d, g, octaves=((1, 1.0), (2, 0.5))):
    """Πείραμα 9: κάθε διάσταση του κλειδιού είναι ένα ομαλό τυχαίο «έδαφος» πάνω στον τόρο (σαν τον θόρυβο του Minecraft).
    Άθροισμα περιοδικών κυμάτων χαμηλής συχνότητας με τυχαίο πλάτος και φάση· χωρίς ετικέτες, μόνο ομαλότητα.
    Κανονικοποιημένο ώστε κάθε διάσταση να έχει std 1, όπως ο λευκός θόρυβος."""
    r, c = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    r, c = r.reshape(-1).float(), c.reshape(-1).float()
    out = torch.zeros(side * side, d)
    for f, amp in octaves:
        for fr, fc in ((f, 0), (0, f), (f, f), (f, -f)):
            a = torch.randn(d, generator=g) * amp
            ph = torch.rand(d, generator=g) * 2 * math.pi
            out += a * torch.cos(2 * math.pi * (fr * r + fc * c)[:, None] / side + ph)
    return (out - out.mean(0)) / out.std(0).clamp_min(1e-8)


class SomMoE(nn.Module):
    """Σκέλη B / C1 / C2: ο νικητής βγαίνει από ευκλείδεια απόσταση στα κλειδιά.

    Κλειδιά: buffer, κινούνται ΜΟΝΟ με τον κανόνα SOM (sweep), ποτέ με backprop.
    Experts: backprop. Βάρη forward: h(d) = exp(-d²/2σ²) στον σταυρό, κανονικοποιημένα.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.n = cfg, cfg.side ** 2
        self.experts = Experts(self.n, cfg.d_model, cfg.expert_hidden)
        g = torch.Generator().manual_seed(1234 + cfg.seed)
        keys = smooth_keys(cfg.side, cfg.d_model, g) if cfg.key_init == "smooth" else torch.randn(self.n, cfg.d_model, generator=g)
        self.register_buffer("keys", keys * 0.1)
        self.register_buffer("D0", metrics.torus_distances(cfg.side))
        self.register_buffer("cross0", metrics.cross(cfg.side))
        self.register_buffer("D", self.D0.clone())
        self.register_buffer("cross", self.cross0.clone())
        self.register_buffer("ema", torch.full((self.n,), 1.0 / self.n))
        self.sigma, self.eta = 2.5, 0.1
        self.last_x = self.last_win = self.last_ce = None

    def set_permutation(self, perm):
        """Τεστ ανακατέματος: ο expert j μετακομίζει στη θέση perm[j]. Κλειδιά και experts μένουν."""
        perm = perm.to(self.D0.device)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(self.n, device=perm.device)
        self.D = self.D0[perm][:, perm]
        self.cross = inv[self.cross0[perm]]

    def congestion(self):
        if self.cfg.arm == "C1":                 # θολωμένο στον σταυρό
            return self.ema[self.cross].mean(-1)
        return self.ema                          # C2: καθαρός DeSieno

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        dist = torch.cdist(x.detach().float(), self.keys)          # ευκλείδεια
        if self.cfg.arm in ("C1", "C2"):
            # Χωρίς κλίμακα: expert με ομοιόμορφο φορτίο (1/64) δεν αλλάζει,
            # με διπλάσιο φαίνεται β φορές πιο μακριά.
            dist = dist * (1 + self.cfg.beta * (self.n * self.congestion() - 1))
        win = dist.argmin(-1)
        members = self.cross[win]                                   # (N, 5)
        d = self.D[win[:, None], members]
        w = torch.exp(-d.pow(2) / (2 * self.sigma ** 2))
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        y = self.experts(x, members, w)
        if self.training:
            self.last_x, self.last_win = x.detach().float(), win
        return y.reshape(shape), {"winner": win, "members": members}

    @torch.no_grad()
    def som_step(self):
        """Κανόνας SOM σε batch: key_j += η·(Σ_t h_tj x_t / Σ_t h_tj − key_j), πλήρες Gaussian."""
        if self.last_x is None:
            return
        x, win = self.last_x, self.last_win
        h = torch.exp(-self.D[win].pow(2) / (2 * self.sigma ** 2))  # (N, 64)
        if self.cfg.arm == "E" and self.last_ce is not None:        # μεγάλο λάθος = τραβάει πιο δυνατά τα κλειδιά
            h = h * (self.last_ce / self.last_ce.mean().clamp_min(1e-8))[:, None]
        mass = h.sum(0)
        target = (h.T @ x) / mass.clamp_min(1e-8)[:, None]
        self.keys += self.eta * (mass > 1e-8).float()[:, None] * (target - self.keys)
        if self.cfg.arm in ("C1", "C2"):                            # EMA ανά token, παγωμένο στο eval
            decay = (1 - self.cfg.ema_alpha) ** len(win)
            freq = torch.bincount(win, minlength=self.n).float() / len(win)
            self.ema.mul_(decay).add_((1 - decay) * freq)
        self.last_x = self.last_win = self.last_ce = None


class Block(nn.Module):
    def __init__(self, cfg, moe):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        if not moe:
            self.mlp = FFN(cfg.d_model, cfg.ffn)
        elif cfg.arm == "A":
            self.mlp = SoftmaxMoE(cfg)
        else:
            self.mlp = SomMoE(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        y, aux = self.mlp(self.ln2(x))
        return x + y, aux


class Lightstorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.arm in ARMS
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i == cfg.moe_layer) for i in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def moe(self):
        return self.blocks[self.cfg.moe_layer].mlp

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        aux = {}
        for blk in self.blocks:
            x, a = blk(x)
            aux.update(a)
        logits = self.ln_f(x) @ self.tok.weight.T                   # tied embeddings
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
            loss = ce.mean()
            aux["ce"] = loss.detach()
            if self.cfg.arm == "E" and self.training:
                self.moe.last_ce = ce.detach().float()
            if "lb" in aux:
                loss = loss + self.cfg.lb_coef * aux["lb"]
        return logits, loss, aux


def schedule(tokens_seen, tokens_per_epoch, sigma=(2.5, 0.7), eta=(0.1, 0.01)):
    """Δύο φάσεις Kohonen σε tokens: 1η εποχή τάξη (εκθετική πτώση), μετά σύγκλιση (σταθερό)."""
    p = min(tokens_seen / tokens_per_epoch, 1.0)
    return sigma[0] * (sigma[1] / sigma[0]) ** p, eta[0] * (eta[1] / eta[0]) ** p


def count_params(model):
    return sum(p.numel() for p in model.parameters())
