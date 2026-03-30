import tempfile
import unittest
from pathlib import Path

import torch

from abstractor import (
    EpisodeRunner,
    PersistentControlConfig,
    PersistentControlLayer,
    generate_synthetic_episodes,
)


class EpisodeLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PersistentControlConfig(
            hidden_dim=12,
            routing_dim=6,
            num_modes=3,
            rank=4,
            plasticity_lr=0.5,
            lambda_m=0.8,
            default_beta=0.7,
            chunk_size=2,
            rho_min=0.0,
            delta_min=0.0,
        )
        self.controller = PersistentControlLayer(self.config)
        centers, episodes, labels = generate_synthetic_episodes(
            config=self.config,
            num_episodes=2,
            batch_size=3,
            seq_len=5,
            noise_scale=0.15,
            seed=123,
        )
        self.controller.seed_modes_from_hidden_centers(centers)
        self.episodes = episodes
        self.labels = labels

    def test_episode_runner_returns_expected_shapes_and_updates_anchor(self) -> None:
        runner = EpisodeRunner(self.controller)
        anchor_before = self.controller.anchor.clone()

        result = runner.run_episode(self.episodes[0], mode_targets=self.labels[0])

        self.assertEqual(tuple(result.modulated_states.shape), tuple(self.episodes[0].shape))
        self.assertEqual(tuple(result.mode_usage.shape), (self.config.num_modes,))
        self.assertEqual(tuple(result.consolidation_gates.shape), (self.config.num_modes,))
        self.assertIsNotNone(result.episode_loss)
        self.assertIsNotNone(result.episode_loss_delta)
        self.assertTrue(result.performance_gate)
        self.assertFalse(torch.allclose(anchor_before, self.controller.anchor))

    def test_reset_episode_restores_scratch_to_anchor(self) -> None:
        self.controller.reset_episode()
        _ = self.controller(self.episodes[0], adapt=True)
        self.assertFalse(torch.allclose(self.controller.scratch, self.controller.anchor))

        self.controller.reset_episode()
        self.assertTrue(torch.allclose(self.controller.scratch, self.controller.anchor))
        self.assertTrue(torch.allclose(self.controller._routing_trace, torch.zeros_like(self.controller._routing_trace)))

    def test_writeback_is_disabled_when_performance_gate_fails(self) -> None:
        runner = EpisodeRunner(self.controller)
        runner.reference_loss = 0.0
        anchor_before = self.controller.anchor.clone()

        result = runner.run_episode(self.episodes[0], mode_targets=self.labels[0])

        self.assertFalse(result.performance_gate)
        self.assertTrue(torch.allclose(anchor_before, self.controller.anchor))

    def test_save_and_load_round_trip(self) -> None:
        runner = EpisodeRunner(self.controller)
        _ = runner.run_episode(self.episodes[0], mode_targets=self.labels[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.pt"
            self.controller.save_persistent_state(path)

            restored = PersistentControlLayer(self.config)
            restored.load_persistent_state(path)

            self.assertTrue(torch.allclose(self.controller.anchor, restored.anchor))
            self.assertTrue(torch.allclose(self.controller.intervention, restored.intervention))
            self.assertTrue(torch.allclose(self.controller.calibration_mean, restored.calibration_mean))


if __name__ == "__main__":
    unittest.main()
