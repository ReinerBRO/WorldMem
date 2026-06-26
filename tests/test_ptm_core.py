import unittest

import torch

from ptm.losses import FutureTestLoss
from ptm.memory import FutureTestDecoder, PTMWorldMemAdapter, PredictiveTestMemory


class PTMCoreTest(unittest.TestCase):
    def test_memory_decoder_loss_backward(self):
        batch, history, future = 2, 12, 6
        frame_dim, action_dim, memory_dim = 8, 5, 32
        memory = PredictiveTestMemory(
            frame_dim=frame_dim,
            action_dim=action_dim,
            memory_dim=memory_dim,
            num_memory_tokens=4,
            num_layers=1,
            num_heads=4,
        )
        decoder = FutureTestDecoder(
            memory_dim=memory_dim,
            action_dim=action_dim,
            future_embedding_dim=frame_dim,
            max_history_candidates=history,
            num_heads=4,
        )
        history_frames = torch.randn(batch, history, frame_dim)
        past_actions = torch.randn(batch, history, action_dim)
        future_actions = torch.randn(batch, future, action_dim)
        labels = {
            "test_type_id": torch.tensor([1, 3]),
            "returns_to_seen_place": torch.tensor([1.0, 0.0]),
            "matched_history_index": torch.tensor([2, 5]),
            "match_valid": torch.tensor([1.0, 1.0]),
            "landmark_visible": torch.tensor([0.0, 1.0]),
            "object_exists_at_return": torch.tensor([1.0, 1.0]),
        }
        target = torch.randn(batch, frame_dim)
        tokens = memory(history_frames, past_actions)
        predictions = decoder(tokens, future_actions, labels["test_type_id"])
        loss, components = FutureTestLoss()(predictions, labels, target)
        loss.backward()
        self.assertGreater(float(components["total"].detach()), 0.0)
        self.assertEqual(tokens.shape, (batch, 4, memory_dim))
        self.assertIsNotNone(memory.memory_queries.grad)

    def test_worldmem_adapter_shapes(self):
        adapter = PTMWorldMemAdapter(
            memory_dim=16,
            latent_channels=4,
            latent_height=3,
            latent_width=5,
            action_dim=7,
            pose_dim=5,
        )
        out = adapter(torch.randn(2, 6, 16))
        self.assertEqual(out["latent_reference_frames"].shape, (6, 2, 4, 3, 5))
        self.assertEqual(out["reference_actions"].shape, (6, 2, 7))
        self.assertEqual(out["reference_poses"].shape, (6, 2, 5))

    def test_future_test_loss_masks_unrelated_heads(self):
        predictions = {
            "future_embedding": torch.zeros(4, 3),
            "loop_return_logit": torch.zeros(4),
            "match_history_logits": torch.zeros(4, 5),
            "landmark_visible_logit": torch.zeros(4),
            "object_exists_logit": torch.zeros(4),
        }
        labels = {
            "test_type_id": torch.tensor([0, 1, 2, 3]),
            "returns_to_seen_place": torch.ones(4),
            "matched_history_index": torch.zeros(4, dtype=torch.long),
            "match_valid": torch.ones(4),
            "landmark_visible": torch.ones(4),
            "object_exists_at_return": torch.ones(4),
        }
        _loss, components = FutureTestLoss()(predictions, labels, torch.zeros(4, 3))
        self.assertIn("loop_return", components)
        self.assertIn("landmark_visible", components)
        self.assertIn("object_exists", components)
        labels["test_type_id"] = torch.zeros(4, dtype=torch.long)
        _loss, components = FutureTestLoss()(predictions, labels, torch.zeros(4, 3))
        self.assertNotIn("loop_return", components)
        self.assertNotIn("landmark_visible", components)
        self.assertNotIn("object_exists", components)

    def test_matched_history_loss_requires_valid_mask(self):
        predictions = {
            "future_embedding": torch.zeros(2, 3),
            "loop_return_logit": torch.zeros(2),
            "match_history_logits": torch.zeros(2, 5),
            "landmark_visible_logit": torch.zeros(2),
            "object_exists_logit": torch.zeros(2),
        }
        labels = {
            "test_type_id": torch.tensor([1, 2]),
            "returns_to_seen_place": torch.ones(2),
            "matched_history_index": torch.tensor([0, 1]),
            "landmark_visible": torch.ones(2),
            "object_exists_at_return": torch.ones(2),
        }
        with self.assertRaises(KeyError):
            FutureTestLoss()(predictions, labels, torch.zeros(2, 3))

        labels["match_valid"] = torch.zeros(2)
        _loss, components = FutureTestLoss()(predictions, labels, torch.zeros(2, 3))
        self.assertNotIn("matched_history", components)

        labels["match_valid"] = torch.tensor([1.0, 0.0])
        labels["matched_history_index"] = torch.tensor([6, 0])
        with self.assertRaises(ValueError):
            FutureTestLoss()(predictions, labels, torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
