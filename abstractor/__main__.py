from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .persistent_control import (
    EpisodeRunner,
    PersistentControlConfig,
    PersistentControlLayer,
    generate_synthetic_episodes,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the persistent writable control-state episode loop.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--routing-dim", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--noise-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-state", type=Path, default=None)
    parser.add_argument("--load-state", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.set_printoptions(precision=3, sci_mode=False)

    config = PersistentControlConfig(
        hidden_dim=args.hidden_dim,
        routing_dim=args.routing_dim,
        num_modes=args.num_modes,
        rank=args.rank,
    )
    controller = PersistentControlLayer(config)

    mode_centers, episodes, labels = generate_synthetic_episodes(
        config=config,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    if args.load_state is not None and args.load_state.exists():
        controller.load_persistent_state(args.load_state)
    else:
        controller.seed_modes_from_hidden_centers(mode_centers)

    runner = EpisodeRunner(controller)

    for episode_index, (hidden_states, episode_labels) in enumerate(zip(episodes, labels), start=1):
        result = runner.run_episode(hidden_states, mode_targets=episode_labels)
        routed_mode = int(result.mode_usage.argmax().item())
        print(
            f"episode={episode_index} "
            f"labels={episode_labels.tolist()} "
            f"dominant_mode={routed_mode} "
            f"mode_usage={result.mode_usage.tolist()} "
            f"null={result.null_usage:.3f} "
            f"entropy={result.routing_entropy:.3f} "
            f"effective_modes={result.effective_modes:.3f} "
            f"max_anchor_similarity={result.max_anchor_similarity:.3f} "
            f"performance_gate={result.performance_gate} "
            f"loss={result.episode_loss:.3f} "
            f"loss_delta={result.episode_loss_delta:.3f} "
            f"gates={result.consolidation_gates.tolist()} "
            f"displacement={result.displacement.tolist()} "
            f"anchor_shift={result.anchor_shift_radians.tolist()}"
        )

    if args.save_state is not None:
        controller.save_persistent_state(args.save_state)
        print(f"saved_state={args.save_state}")


if __name__ == "__main__":
    main()
