#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pagerank.pagerank_with_numpy."""

import numpy

from pagerank.exception import ConvergenceError
from pagerank.logger import get_logger


class PageRankNumpy:
    """PageRank algorithm with Numpy.

    The algorithm uses the power method to estimate the page-ranks.

    Parameters
    ----------
    damping_parameter:
        The damping parameter is often denoted as d.

    tolerance:
        Once the max update is less than this tolerance, the iterative
        process terminates.

    max_iter:
        The maximum number of iteration.

    verbose:
        Whether to print out the detailed logs.
    """

    def __init__(
        self,
        damping_parameter: float = 0.85,
        tolerance: float = 1e-4,
        max_iter: int = 1000,
        verbose: bool = True,
    ) -> None:
        self.damping_parameter = damping_parameter
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.logger = get_logger("PageRankNumpy", "DEBUG" if verbose else "WARNING")

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self) -> str:
        return (
            "PageRank algorithm with numpy: damping_parameter = %.3f"
            % self.damping_parameter
        )

    def score(self, adjacency_matrix: numpy.ndarray) -> numpy.ndarray:
        """Score the websites.

        The i-th row j-th column in the adjacency matrix indicates whether page
        j links to page i.
        """
        n_webpages = adjacency_matrix.shape[0]
        ones = numpy.ones((n_webpages, 1))

        outlinks = ones.T.dot(adjacency_matrix).flatten()
        weights = numpy.diag(1.0 / outlinks)

        transition_matrix = (
            (1 - self.damping_parameter) * ones.dot(ones.T) / n_webpages
        ) + (self.damping_parameter * adjacency_matrix.dot(weights))

        pagerank = ones.flatten()

        converged = False
        for i in range(self.max_iter):
            new_pagerank = transition_matrix.dot(pagerank)
            new_pagerank = n_webpages * new_pagerank / ones.T.dot(new_pagerank)

            max_update = max(abs(new_pagerank - pagerank))
            pagerank = new_pagerank

            if max_update < self.tolerance:
                converged = True
                self.logger.info("The algorithm converged after %i iterations.", i)
                break

        if not converged:
            raise ConvergenceError(
                "The algorithm did not converge within %i iterations." % self.max_iter
            )

        return pagerank
