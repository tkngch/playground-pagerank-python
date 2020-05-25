#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests.test_pagerank_with_torch."""

import torch

from pagerank import PageRankTorch


def test() -> None:
    """Test if the PageRank algorithm returns the correct scores.

    The correct scores are taken from the ESL book, page 578.
    """
    damping_parameter = 0.85
    algorithm = PageRankTorch(damping_parameter, verbose=False)

    adjacency_matrix = torch.tensor(
        [[0.0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]]
    )
    pagerank = algorithm.score(adjacency_matrix)

    expected = torch.tensor([[1.49, 0.78, 1.58, 0.15]]).T
    assert torch.allclose(_round_tensor(pagerank, 2), expected)


def _round_tensor(tensor: torch.Tensor, n_decimal_point: int) -> torch.Tensor:
    multiplier = 10 ** n_decimal_point
    return (tensor * multiplier).round() / multiplier


def test_properties() -> None:
    """Test the properties of PageRankNumpy class."""
    algorithm = PageRankTorch()

    assert isinstance(algorithm.__repr__(), str)
    assert isinstance(algorithm.__str__(), str)
