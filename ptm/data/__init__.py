"""Data generation, schema, and loading utilities for PTM."""

from .schemas import (
    ACTION_KEYS,
    TEST_TYPE_TO_ID,
    EventRecord,
    PoseRecord,
    TestRecord,
    compact_minedojo_action_to_oasis,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    "ACTION_KEYS",
    "TEST_TYPE_TO_ID",
    "EventRecord",
    "PoseRecord",
    "TestRecord",
    "compact_minedojo_action_to_oasis",
    "read_jsonl",
    "write_jsonl",
    "PTMWorldMemDataset",
]


def __getattr__(name: str):
    if name == "PTMWorldMemDataset":
        from .worldmem_dataset import PTMWorldMemDataset

        return PTMWorldMemDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
