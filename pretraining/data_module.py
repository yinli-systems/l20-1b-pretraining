#!/usr/bin/env python3
"""Resumable weighted LitData mixture for the 20B-token run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import DataLoader

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class HighQualityEnglish(DataModule):
    data_path: str | Path = Path("/home/hhai/pretrain/data/packed")
    seed: int = 42
    num_workers: int = 2
    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2049)

    SOURCES = ("web", "dclm", "math", "code")
    # Evidence-backed short-horizon foundation mix: 85% English web split
    # evenly between FineWeb-Edu and DCLM, 3% high-quality math, 12% code.
    TRAIN_WEIGHTS = (0.425, 0.425, 0.03, 0.12)

    def connect(
        self, tokenizer: Tokenizer | None = None, batch_size: int = 1, max_seq_length: int | None = None
    ) -> None:
        if max_seq_length is None:
            raise ValueError("max_seq_length is required")
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1

    def prepare_data(self) -> None:
        missing = []
        for source in self.SOURCES:
            for split in ("train", "val"):
                path = Path(self.data_path) / source / split / "index.json"
                if not path.is_file():
                    missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"missing packed datasets: {missing}")

    def _combined(self, split: str, shuffle: bool):
        from litdata.streaming import CombinedStreamingDataset, StreamingDataset, TokensLoader

        datasets = [
            StreamingDataset(
                input_dir=str(Path(self.data_path) / source / split),
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=shuffle,
                drop_last=True,
                seed=self.seed + index,
            )
            for index, source in enumerate(self.SOURCES)
        ]
        return CombinedStreamingDataset(
            datasets=datasets,
            seed=self.seed,
            weights=self.TRAIN_WEIGHTS,
            iterate_over_all=False,
        )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader

        return StreamingDataLoader(
            self._combined("train", shuffle=True),
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader

        return StreamingDataLoader(
            self._combined("val", shuffle=False),
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )
