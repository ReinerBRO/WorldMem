import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader

from ptm.data.minedojo_generator import generate
from ptm.data.npz_cache import _history_match_label, build_npz_cache
from ptm.data.ptm_dataset import PTMDataset, collate_ptm_batch
from ptm.data.verify_dataset import find_episodes, verify_episode
from ptm.data.worldmem_dataset import PTMWorldMemDataset, collate_worldmem_ptm


class PTMDatasetTest(unittest.TestCase):
    def test_history_match_label_is_valid_only(self):
        index, valid, candidates = _history_match_label(
            matched_t=397,
            memory_indices=[1, 41, 81, 150, 300, 450, 599, 600],
            main_indices=[597, 598, 599, 600, 601, 602, 603, 604],
            history_context_end_index=3,
            max_history_candidates=708,
        )
        self.assertEqual(index, 0)
        self.assertEqual(valid, 0)
        self.assertEqual(candidates, [1, 41, 81, 150, 300, 450, 599, 600, 597, 598])

        index, valid, _candidates = _history_match_label(
            matched_t=599,
            memory_indices=[1, 41, 81, 150, 300, 450, 599, 600],
            main_indices=[597, 598, 599, 600, 601, 602, 603, 604],
            history_context_end_index=3,
            max_history_candidates=708,
        )
        self.assertEqual(index, 6)
        self.assertEqual(valid, 1)

    def test_mock_generation_verify_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "stage0"
            args = Namespace(
                out=str(root),
                num_episodes=6,
                episode_offset=0,
                schedule_total=None,
                skip_existing=False,
                frames_per_episode=32,
                families="loop_return_pos,loop_return_neg,landmark_revisit_pos,landmark_revisit_neg,object_persistence_pos,object_persistence_neg",
                seed=7,
                split="train",
                backend="mock",
                height=32,
                width=32,
                fps=10.0,
                env_type="plains",
                weather="clear",
                frame_storage="npz",
                history_length=8,
                future_length=6,
                test_stride=4,
                lock_stale_seconds=7200,
                episode_locks=False,
                atomic_write=True,
                continue_on_error=False,
                episode_retries=0,
                episode_retry_sleep=0.0,
                episode_retry_seed_stride=1000003,
                episode_timeout_seconds=0.0,
                reuse_env=False,
                fast_reset=True,
                fast_reset_random_teleport_range=200,
            )
            generate(args)
            val_args = Namespace(**vars(args))
            val_args.split = "val"
            val_args.num_episodes = 2
            generate(val_args)
            episodes = find_episodes(root / "train")
            self.assertEqual(len(episodes), 6)
            for episode in episodes:
                self.assertEqual(verify_episode(episode), [])

            dataset = PTMDataset(root, split="train", history_length=8, future_length=6, resolution=32)
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_ptm_batch)
            batch = next(iter(loader))
            self.assertEqual(batch["history_frames"].shape[1:], (8, 3, 32, 32))
            self.assertEqual(batch["future_actions"].shape[1], 6)
            self.assertIn("matched_history_index", batch["memory_labels"])
            self.assertIn("match_valid", batch["memory_labels"])

            cfg = SimpleNamespace(
                save_dir=str(root),
                resolution=32,
                n_frames=10,
                memory_condition_length=4,
                ptm_context_length=4,
                ptm_future_length=6,
                future_length=6,
                max_history_candidates=16,
                frame_skip=1,
            )
            worldmem_dataset = PTMWorldMemDataset(cfg, split="training")
            item = worldmem_dataset[0]
            self.assertEqual(item["video"].shape, (14, 3, 32, 32))
            self.assertEqual(item["actions"].shape, (14, 25))
            self.assertEqual(item["future_actions"].shape, (6, 25))
            self.assertTrue(bool(item["has_reference_tail"]))
            self.assertIn("test_type_id", item["memory_labels"])
            self.assertIn("match_valid", item["memory_labels"])
            self.assertGreaterEqual(int(item["query_index_in_video"]), int(item["context_length"]) - 1)
            self.assertGreaterEqual(int(item["target_index_in_video"]), int(item["query_index_in_video"]))
            worldmem_loader = DataLoader(worldmem_dataset, batch_size=2, collate_fn=collate_worldmem_ptm)
            worldmem_batch = next(iter(worldmem_loader))
            self.assertIn("test_type_id", worldmem_batch["memory_labels"])
            self.assertEqual(worldmem_batch["memory_labels"]["test_type_id"].shape, (2,))
            self.assertIn("match_valid", worldmem_batch["memory_labels"])
            self.assertEqual(worldmem_batch["memory_labels"]["match_valid"].shape, (2,))
            self.assertEqual(worldmem_batch["query_index_in_video"].shape, (2,))
            self.assertEqual(worldmem_batch["context_length"].shape, (2,))
            self.assertTrue(bool(worldmem_batch["has_reference_tail"].all()))

            cache_root = Path(tmp) / "cache"
            build_npz_cache(
                data_root=root,
                out_dir=cache_root,
                splits=["train"],
                workers=1,
                height=32,
                width=32,
                context_length=4,
                future_length=6,
                memory_condition_length=4,
                max_history_candidates=16,
            )
            cached_cfg = SimpleNamespace(**vars(cfg), npz_cache_dir=str(cache_root))
            cached_dataset = PTMWorldMemDataset(cached_cfg, split="training")
            cached_item = cached_dataset[0]
            self.assertEqual(cached_item["video"].shape, item["video"].shape)
            self.assertEqual(cached_item["actions"].shape, item["actions"].shape)
            self.assertEqual(cached_item["target_frames"].shape, item["target_frames"].shape)
            self.assertTrue(bool(cached_item["has_reference_tail"]))
            self.assertIn("match_valid", cached_item["memory_labels"])

            val_cfg = SimpleNamespace(
                **vars(cfg),
                npz_cache_dir=str(cache_root),
                n_frames_valid=12,
                ptm_context_length_valid=8,
                ptm_future_length_valid=4,
            )
            val_dataset = PTMWorldMemDataset(val_cfg, split="validation")
            val_item = val_dataset[0]
            self.assertEqual(val_item["video"].shape, (12, 3, 32, 32))
            self.assertEqual(val_item["actions"].shape, (12, 25))
            self.assertEqual(val_item["future_actions"].shape, (4, 25))
            self.assertEqual(int(val_item["context_length"]), 8)
            self.assertFalse(bool(val_item["has_reference_tail"]))
            self.assertGreaterEqual(int(val_item["query_index_in_video"]), 7)


if __name__ == "__main__":
    unittest.main()
