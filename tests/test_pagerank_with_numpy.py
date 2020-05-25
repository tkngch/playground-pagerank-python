#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tests.test_pagerank_with_numpy."""

import numpy
from numpy.testing import assert_allclose

from pagerank import PageRankNumpy


def test() -> None:
    """Test if the PageRank algorithm returns the correct scores.

    The correct scores are taken from the ESL book, page 578.
    """
    damping_parameter = 0.85
    algorithm = PageRankNumpy(damping_parameter, verbose=False)

    adjacency_matrix = numpy.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]]
    )
    pagerank = algorithm.score(adjacency_matrix)

    expected = numpy.array([1.49, 0.78, 1.58, 0.15])
    assert_allclose(numpy.round(pagerank, 2), expected)


def test_properties() -> None:
    """Test the properties of PageRankNumpy class."""
    algorithm = PageRankNumpy()

    assert isinstance(algorithm.__repr__(), str)
    assert isinstance(algorithm.__str__(), str)
