"""Retrieval for the visual graph algorithm."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import cv2
import numpy as np

from .models import (
    Config,
    FeatureRecord,
    ImageKey,
    RetrievalCandidate,
    StereoPair,
)


class IncrementalSIFTRetriever:
    """Replaceable query-before-add descriptor voting interface.

    Each train matrix corresponds to exactly one historical pair. Temporal
    exclusion uses stream positions, not possibly discontinuous frame numbers.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=cfg.FLANN_TREES), dict(checks=cfg.FLANN_CHECKS)
        )
        self.history: list[tuple[int, int]] = []
        self.total_descriptors = 0
        self.last_position = -1

    def describe_pair(
        self, pair: StereoPair, feature: Callable[[ImageKey], FeatureRecord]
    ) -> np.ndarray:
        matrices = []
        for key in (pair.left_key, pair.right_key):
            record = feature(key)
            indices = np.argsort(-record.responses, kind="stable")[
                : self.cfg.RETRIEVAL_DESCRIPTORS_PER_IMAGE
            ]
            matrices.append(record.descriptors[indices])
        return np.ascontiguousarray(np.vstack(matrices), dtype=np.float32)

    def query(self, descriptors: np.ndarray, position: int) -> list[RetrievalCandidate]:
        assert position > self.last_position, "Retrieval must query only history"
        if self.total_descriptors < 2 or len(descriptors) == 0:
            return []
        votes: Counter = Counter()
        weighted: defaultdict = defaultdict(float)
        for matches in self.matcher.knnMatch(descriptors, k=2):
            if len(matches) != 2:
                continue
            m, n = matches
            if m.distance >= self.cfg.RETRIEVAL_RATIO * n.distance:
                continue
            _, historical_position = self.history[m.imgIdx]
            assert historical_position < position
            if position - historical_position < self.cfg.RETRIEVAL_TEMPORAL_EXCLUSION:
                continue
            votes[m.imgIdx] += 1
            weighted[m.imgIdx] += 1.0 / (m.distance + 1e-6)
        ranked = sorted(votes, key=lambda idx: (-votes[idx], -weighted[idx], idx))
        return [
            RetrievalCandidate(
                self.history[idx][0],
                self.history[idx][1],
                votes[idx],
                votes[idx] / len(descriptors),
                weighted[idx] / len(descriptors),
            )
            for idx in ranked[: self.cfg.RETRIEVAL_TOP_K]
        ]

    def add(self, frame: int, position: int, descriptors: np.ndarray) -> None:
        assert position > self.last_position
        self.last_position = position
        if len(descriptors):
            self.matcher.add([descriptors])
            self.matcher.train()
            self.history.append((frame, position))
            self.total_descriptors += len(descriptors)
