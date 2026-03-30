from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _normalize(value: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    return F.normalize(value, dim=dim, eps=eps)


@dataclass(frozen=True)
class PersistentControlConfig:
    hidden_dim: int = 32
    routing_dim: int = 16
    num_modes: int = 4
    rank: int = 8
    alpha: float = 0.35
    plasticity_lr: float = 0.25
    lambda_m: float = 0.9
    theta_bound: float = 0.45
    chunk_size: int = 4
    rho_min: float = 0.12
    delta_min: float = 0.005
    default_beta: float = 0.25
    calibration_momentum: float = 0.9
    eps: float = 1e-6


@dataclass
class EpisodeResult:
    modulated_states: Tensor
    mode_usage: Tensor
    null_usage: float
    routing_entropy: float
    effective_modes: float
    max_anchor_similarity: float
    displacement: Tensor
    consolidation_gates: Tensor
    performance_gate: bool
    anchor_shift_radians: Tensor
    episode_loss: Optional[float]
    episode_loss_delta: Optional[float]


class PersistentControlLayer(nn.Module):
    def __init__(self, config: PersistentControlConfig) -> None:
        super().__init__()
        if config.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.config = config

        self.routing_projection = nn.Linear(config.hidden_dim, config.routing_dim, bias=False)
        self.down_project = nn.Linear(config.hidden_dim, config.rank, bias=False)
        self.gate_projection = nn.Linear(config.hidden_dim, config.rank, bias=False)
        self.up_project = nn.Linear(config.rank, config.hidden_dim, bias=False)

        self.register_buffer(
            "anchor",
            _normalize(torch.randn(config.num_modes, config.routing_dim), dim=-1, eps=config.eps),
        )
        self.register_buffer("scratch", self.anchor.clone())
        self.register_buffer(
            "intervention",
            _normalize(torch.randn(config.num_modes, config.hidden_dim), dim=-1, eps=config.eps),
        )
        self.register_buffer("threshold", torch.zeros(config.num_modes))
        self.register_buffer("beta", torch.full((config.num_modes,), config.default_beta))
        self.register_buffer("calibration_mean", torch.zeros(config.num_modes))
        self.register_buffer("calibration_std", torch.ones(config.num_modes))

        self.register_buffer(
            "_episode_usage_accumulator",
            torch.zeros(config.num_modes),
            persistent=False,
        )
        self.register_buffer("_episode_null_accumulator", torch.zeros(1), persistent=False)
        self.register_buffer("_episode_entropy_accumulator", torch.zeros(1), persistent=False)
        self.register_buffer("_episode_steps", torch.zeros(1), persistent=False)
        self.register_buffer(
            "_routing_trace",
            torch.zeros(config.num_modes, config.routing_dim),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.routing_projection.weight)
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.xavier_uniform_(self.gate_projection.weight)
        nn.init.xavier_uniform_(self.up_project.weight)

    @torch.no_grad()
    def reset_episode(self) -> None:
        self.scratch.copy_(self.anchor)
        self._episode_usage_accumulator.zero_()
        self._episode_null_accumulator.zero_()
        self._episode_entropy_accumulator.zero_()
        self._episode_steps.zero_()
        self._routing_trace.zero_()

    @torch.no_grad()
    def seed_modes_from_hidden_centers(self, centers: Tensor) -> None:
        projected = self.routing_projection(centers)
        projected = _normalize(projected, dim=-1, eps=self.config.eps)
        count = min(projected.shape[0], self.anchor.shape[0])
        self.anchor[:count].copy_(projected[:count])
        self.scratch.copy_(self.anchor)

    def forward(self, hidden_states: Tensor, adapt: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
        added_batch = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            added_batch = True

        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must have shape [batch, seq, hidden] or [seq, hidden]")

        batch_size, seq_len, _ = hidden_states.shape
        intervention_gate = 2.0 * torch.sigmoid(self.gate_projection(self.intervention)) - 1.0

        modulated_steps: List[Tensor] = []
        routing_steps: List[Tensor] = []
        score_steps: List[Tensor] = []
        shift_norms: List[Tensor] = []

        for step_index in range(seq_len):
            step_hidden = hidden_states[:, step_index, :]
            z = self.routing_projection(step_hidden)
            z_norm = _normalize(z, dim=-1, eps=self.config.eps)
            scratch_norm = _normalize(self.scratch, dim=-1, eps=self.config.eps)

            scores = torch.einsum("br,kr->bk", z_norm, scratch_norm)
            standardized = (
                (scores - self.calibration_mean.view(1, -1))
                / self.calibration_std.view(1, -1).clamp_min(self.config.eps)
            ) - self.threshold.view(1, -1)

            null_logits = torch.zeros(
                batch_size,
                1,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            routing_distribution = torch.softmax(
                torch.cat([standardized, null_logits], dim=-1),
                dim=-1,
            )
            mode_probs = routing_distribution[..., : self.config.num_modes]

            low_rank_base = self.down_project(step_hidden)
            weighted_gate = torch.einsum("bk,kr->br", mode_probs, intervention_gate)
            delta = self.config.alpha * self.up_project(low_rank_base * weighted_gate)

            modulated_steps.append(step_hidden + delta)
            routing_steps.append(routing_distribution)
            score_steps.append(scores)
            shift_norms.append(delta.norm(dim=-1).mean())

            routing_entropy = -(
                routing_distribution.clamp_min(self.config.eps)
                * routing_distribution.clamp_min(self.config.eps).log()
            ).sum(dim=-1).mean()

            with torch.no_grad():
                self._episode_usage_accumulator.add_(mode_probs.mean(dim=0))
                self._episode_null_accumulator.add_(routing_distribution[..., -1].mean())
                self._episode_entropy_accumulator.add_(routing_entropy)
                self._episode_steps.add_(1.0)

                if adapt:
                    self._accumulate_routing_trace(z_norm, mode_probs)
                    if (step_index + 1) % self.config.chunk_size == 0 or step_index == seq_len - 1:
                        self._apply_scratch_update()

        routing_distribution = torch.stack(routing_steps, dim=1)
        mode_probs = routing_distribution[..., : self.config.num_modes]
        scores = torch.stack(score_steps, dim=1)
        modulated = torch.stack(modulated_steps, dim=1)

        with torch.no_grad():
            self._update_calibration(scores)

        mode_usage = mode_probs.mean(dim=(0, 1))
        null_usage = routing_distribution[..., -1].mean()
        routing_entropy = -(
            routing_distribution.clamp_min(self.config.eps)
            * routing_distribution.clamp_min(self.config.eps).log()
        ).sum(dim=-1).mean()

        metrics = {
            "mode_usage": mode_usage.detach().clone(),
            "null_usage": null_usage.detach().clone(),
            "routing_entropy": routing_entropy.detach().clone(),
            "mean_resonance": scores.mean().detach().clone(),
            "mean_shift_norm": torch.stack(shift_norms).mean().detach().clone(),
            "mode_probs": mode_probs.detach().clone(),
        }

        if added_batch:
            modulated = modulated.squeeze(0)
        return modulated, metrics

    @torch.no_grad()
    def consolidate_episode(
        self,
        episode_loss_delta: Optional[float] = None,
        allow_writeback: bool = True,
    ) -> Dict[str, Tensor]:
        if self._episode_steps.item() == 0.0:
            raise RuntimeError("consolidate_episode requires at least one forward pass in the episode")

        mean_usage = self._episode_usage_accumulator / self._episode_steps.clamp_min(1.0)
        mean_null = self._episode_null_accumulator / self._episode_steps.clamp_min(1.0)
        mean_entropy = self._episode_entropy_accumulator / self._episode_steps.clamp_min(1.0)

        cosine = (self.anchor * self.scratch).sum(dim=-1).clamp(-1.0, 1.0)
        displacement = 1.0 - cosine

        usage_gate = (mean_usage > self.config.rho_min).to(self.anchor.dtype)
        displacement_gate = (displacement > self.config.delta_min).to(self.anchor.dtype)
        performance_gate_value = episode_loss_delta is None or episode_loss_delta < 0.0
        performance_gate = torch.full_like(
            mean_usage,
            1.0 if performance_gate_value else 0.0,
        )

        gates = usage_gate * displacement_gate * performance_gate
        if not allow_writeback:
            gates.zero_()

        previous_anchor = self.anchor.clone()
        mix = (self.beta * gates).unsqueeze(-1)
        updated_anchor = _normalize(
            (1.0 - mix) * self.anchor + mix * self.scratch,
            dim=-1,
            eps=self.config.eps,
        )
        self.anchor.copy_(updated_anchor)
        self.scratch.copy_(self.anchor)

        cosine_shift = (previous_anchor * self.anchor).sum(dim=-1).clamp(-1.0, 1.0)
        anchor_shift_radians = torch.arccos(cosine_shift)
        effective_modes = self._effective_modes(mean_usage)
        max_anchor_similarity = self._max_off_diagonal_similarity(self.anchor)

        summary = {
            "mode_usage": mean_usage.detach().clone(),
            "null_usage": mean_null.squeeze(0).detach().clone(),
            "routing_entropy": mean_entropy.squeeze(0).detach().clone(),
            "effective_modes": torch.tensor(
                effective_modes,
                device=self.anchor.device,
                dtype=self.anchor.dtype,
            ),
            "max_anchor_similarity": torch.tensor(
                max_anchor_similarity,
                device=self.anchor.device,
                dtype=self.anchor.dtype,
            ),
            "displacement": displacement.detach().clone(),
            "consolidation_gates": gates.detach().clone(),
            "usage_gate": usage_gate.detach().clone(),
            "displacement_gate": displacement_gate.detach().clone(),
            "performance_gate": torch.tensor(
                1.0 if performance_gate_value else 0.0,
                device=self.anchor.device,
                dtype=self.anchor.dtype,
            ),
            "anchor_shift_radians": anchor_shift_radians.detach().clone(),
        }

        self._episode_usage_accumulator.zero_()
        self._episode_null_accumulator.zero_()
        self._episode_entropy_accumulator.zero_()
        self._episode_steps.zero_()
        self._routing_trace.zero_()
        return summary

    @torch.no_grad()
    def save_persistent_state(self, path: Union[str, Path]) -> None:
        payload = {
            "config": self.config.__dict__,
            "anchor": self.anchor.detach().cpu(),
            "intervention": self.intervention.detach().cpu(),
            "threshold": self.threshold.detach().cpu(),
            "beta": self.beta.detach().cpu(),
            "calibration_mean": self.calibration_mean.detach().cpu(),
            "calibration_std": self.calibration_std.detach().cpu(),
        }
        torch.save(payload, Path(path))

    @torch.no_grad()
    def load_persistent_state(
        self,
        path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> None:
        payload = torch.load(Path(path), map_location=map_location)
        self.anchor.copy_(payload["anchor"].to(self.anchor.device))
        self.scratch.copy_(self.anchor)
        self.intervention.copy_(payload["intervention"].to(self.intervention.device))
        self.threshold.copy_(payload["threshold"].to(self.threshold.device))
        self.beta.copy_(payload["beta"].to(self.beta.device))
        self.calibration_mean.copy_(payload["calibration_mean"].to(self.calibration_mean.device))
        self.calibration_std.copy_(payload["calibration_std"].to(self.calibration_std.device))

    @torch.no_grad()
    def _update_calibration(self, scores: Tensor) -> None:
        batch_mean = scores.mean(dim=(0, 1))
        batch_std = scores.std(dim=(0, 1), unbiased=False).clamp_min(self.config.eps)
        keep = self.config.calibration_momentum
        update = 1.0 - keep
        self.calibration_mean.mul_(keep).add_(batch_mean * update)
        self.calibration_std.mul_(keep).add_(batch_std * update)

    @torch.no_grad()
    def _accumulate_routing_trace(self, z_norm: Tensor, mode_probs: Tensor) -> None:
        step_trace = torch.einsum("bk,br->kr", mode_probs, z_norm) / max(z_norm.shape[0], 1)
        self._routing_trace.mul_(self.config.lambda_m).add_(step_trace)

    @torch.no_grad()
    def _apply_scratch_update(self) -> None:
        scratch_norm = _normalize(self.scratch, dim=-1, eps=self.config.eps)
        tangent_update = self._routing_trace - (
            (self._routing_trace * scratch_norm).sum(dim=-1, keepdim=True) * scratch_norm
        )
        candidate = _normalize(
            self.scratch + self.config.plasticity_lr * tangent_update,
            dim=-1,
            eps=self.config.eps,
        )
        self.scratch.copy_(self._project_to_cone(self.anchor, candidate))

    def _project_to_cone(self, anchor: Tensor, candidate: Tensor) -> Tensor:
        if self.config.routing_dim == 1:
            return anchor.clone()

        anchor_norm = _normalize(anchor, dim=-1, eps=self.config.eps)
        candidate_norm = _normalize(candidate, dim=-1, eps=self.config.eps)

        cos_bound = math.cos(self.config.theta_bound)
        sin_bound = math.sin(self.config.theta_bound)

        cosine = (anchor_norm * candidate_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        within_bound = cosine >= cos_bound

        orthogonal = candidate_norm - cosine * anchor_norm
        orthogonal_norm = orthogonal.norm(dim=-1, keepdim=True)
        fallback = self._fallback_orthogonal(anchor_norm)
        orthogonal_dir = torch.where(
            orthogonal_norm > self.config.eps,
            orthogonal / orthogonal_norm.clamp_min(self.config.eps),
            fallback,
        )

        bounded = _normalize(
            cos_bound * anchor_norm + sin_bound * orthogonal_dir,
            dim=-1,
            eps=self.config.eps,
        )
        return torch.where(within_bound, candidate_norm, bounded)

    def _fallback_orthogonal(self, anchor: Tensor) -> Tensor:
        basis = torch.zeros_like(anchor)
        basis[:, 0] = 1.0
        if anchor.shape[-1] > 1:
            use_second_basis = anchor[:, 0].abs() > 0.9
            basis[use_second_basis] = 0.0
            basis[use_second_basis, 1] = 1.0

        orthogonal = basis - (basis * anchor).sum(dim=-1, keepdim=True) * anchor
        return _normalize(orthogonal, dim=-1, eps=self.config.eps)

    def _effective_modes(self, mean_usage: Tensor) -> float:
        active_mass = mean_usage.sum().item()
        if active_mass <= self.config.eps:
            return 0.0

        normalized_usage = mean_usage / mean_usage.sum().clamp_min(self.config.eps)
        entropy = -(
            normalized_usage.clamp_min(self.config.eps)
            * normalized_usage.clamp_min(self.config.eps).log()
        ).sum()
        return float(torch.exp(entropy).item())

    def _max_off_diagonal_similarity(self, anchor: Tensor) -> float:
        if anchor.shape[0] <= 1:
            return 0.0

        similarity = anchor @ anchor.T
        mask = ~torch.eye(anchor.shape[0], dtype=torch.bool, device=anchor.device)
        return float(similarity.masked_select(mask).max().item())


class EpisodeRunner:
    def __init__(self, controller: PersistentControlLayer, reference_decay: float = 0.9) -> None:
        self.controller = controller
        self.reference_loss: Optional[float] = None
        self.reference_decay = reference_decay

    @torch.no_grad()
    def run_episode(
        self,
        hidden_states: Tensor,
        mode_targets: Optional[Tensor] = None,
        allow_writeback: bool = True,
    ) -> EpisodeResult:
        self.controller.reset_episode()
        modulated, metrics = self.controller(hidden_states, adapt=True)

        episode_loss: Optional[float] = None
        episode_loss_delta: Optional[float] = None
        if mode_targets is not None:
            episode_loss = self._episode_routing_loss(metrics["mode_probs"], mode_targets)
            baseline = self.reference_loss if self.reference_loss is not None else math.log(self.controller.config.num_modes)
            episode_loss_delta = episode_loss - baseline
            if self.reference_loss is None:
                self.reference_loss = episode_loss
            else:
                self.reference_loss = (
                    self.reference_decay * self.reference_loss
                    + (1.0 - self.reference_decay) * episode_loss
                )

        summary = self.controller.consolidate_episode(
            episode_loss_delta=episode_loss_delta,
            allow_writeback=allow_writeback,
        )
        return EpisodeResult(
            modulated_states=modulated.detach().clone(),
            mode_usage=summary["mode_usage"],
            null_usage=float(summary["null_usage"].item()),
            routing_entropy=float(summary["routing_entropy"].item()),
            effective_modes=float(summary["effective_modes"].item()),
            max_anchor_similarity=float(summary["max_anchor_similarity"].item()),
            displacement=summary["displacement"],
            consolidation_gates=summary["consolidation_gates"],
            performance_gate=bool(summary["performance_gate"].item()),
            anchor_shift_radians=summary["anchor_shift_radians"],
            episode_loss=episode_loss,
            episode_loss_delta=episode_loss_delta,
        )

    @torch.no_grad()
    def run(self, episodes: Sequence[Tensor]) -> List[EpisodeResult]:
        return [self.run_episode(episode) for episode in episodes]

    def _episode_routing_loss(self, mode_probs: Tensor, mode_targets: Tensor) -> float:
        if mode_targets.dim() == 1:
            expanded_targets = mode_targets.unsqueeze(1).expand(-1, mode_probs.shape[1])
        elif mode_targets.dim() == 2:
            expanded_targets = mode_targets
        else:
            raise ValueError("mode_targets must have shape [batch] or [batch, seq]")

        selected = mode_probs.clamp_min(self.controller.config.eps).gather(
            -1,
            expanded_targets.unsqueeze(-1),
        )
        return float((-selected.log()).mean().item())


def generate_synthetic_episodes(
    config: PersistentControlConfig,
    num_episodes: int,
    batch_size: int,
    seq_len: int,
    noise_scale: float = 0.2,
    seed: int = 0,
) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    mode_centers = _normalize(
        torch.randn(config.num_modes, config.hidden_dim, generator=generator),
        dim=-1,
        eps=config.eps,
    )

    episodes: List[Tensor] = []
    labels: List[Tensor] = []
    for _ in range(num_episodes):
        episode_labels = torch.randint(0, config.num_modes, (batch_size,), generator=generator)
        center = mode_centers[episode_labels].unsqueeze(1).expand(-1, seq_len, -1)
        noise = noise_scale * torch.randn(batch_size, seq_len, config.hidden_dim, generator=generator)
        episodes.append(center + noise)
        labels.append(episode_labels)

    return mode_centers, episodes, labels
