"""Preprocessing must preserve size and fit only the supplied train features."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..data import (
    center_images,
    fit_routing,
    pooled_features,
    stratified_indices,
    validate_targets,
)


def test_centering_preserves_foreground_and_removes_translation_without_resizing():
    images = torch.zeros(3, 32, 32, dtype=torch.uint8)
    images[0, 0:4, 0:6] = 1
    images[1, 28:32, 26:32] = 1
    images[2, 9:13, 11:17] = 1
    centered = center_images(images)
    assert torch.equal(centered[0], centered[1])
    assert torch.equal(centered[1], centered[2])
    assert centered.sum().item() == images.sum().item() == 72
    features = pooled_features(images.numpy())
    assert features.shape == (3, 40)
    np.testing.assert_array_equal(features[0], features[2])
    with pytest.raises(ValueError, match="Empty"):
        center_images(torch.zeros_like(images))


def test_variance_rank_ties_and_round_robin():
    # Constant training pixels must retain stable source-index order, irrespective
    # of any differently distributed validation pixels.
    training = np.zeros((10, 40), dtype=np.float32)
    permutation, variances = fit_routing(training)
    np.testing.assert_array_equal(
        permutation.reshape(10, 4), np.arange(40).reshape(4, 10).T
    )
    assert np.all(variances == 0)
    training[0, 39] = 1
    permutation, _ = fit_routing(training)
    assert permutation[0] == 39
    assert set(permutation) == set(range(40))


def test_subset_and_target_contract():
    concepts = np.array(
        [(shape, scale) for shape in range(3) for scale in range(6)] * 4
    )
    selected = stratified_indices(concepts, 36, seed=4)
    assert len(np.unique(selected)) == 36
    assert len(np.unique(concepts[selected], axis=0)) == 18
    labels = ((concepts[:, 0] == 2) ^ (concepts[:, 1] > 2)).astype(np.int64)
    validate_targets(concepts, labels)
    with pytest.raises(ValueError, match="compact_c"):
        validate_targets(concepts, 1 - labels)
