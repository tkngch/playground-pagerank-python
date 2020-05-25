#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pagerank.pagerank_with_torch."""

import torch

from pagerank.exception import ConvergenceError
from pagerank.logger import get_logger


class PageRankTorch:
    """PageRank algorithm with Torch.

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
        self.logger = get_logger("PageRankTorch", "DEBUG" if verbose else "WARNING")

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self) -> str:
        return (
            "PageRank algorithm with torch: damping_parameter = %.3f"
            % self.damping_parameter
        )

    def score(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Score the websites.

        The i-th row j-th column in the adjacency matrix indicates whether page
        j links to page i.
        """
        n_webpages = adjacency_matrix.size(0)
        ones = torch.ones((n_webpages, 1))

        outlinks = ones.T.mm(adjacency_matrix).view((n_webpages,))
        weights = torch.diag(1.0 / outlinks)

        transition_matrix = (
            (1 - self.damping_parameter) * ones.mm(ones.T) / n_webpages
        ) + (self.damping_parameter * adjacency_matrix.mm(weights))

        pagerank = torch.ones((n_webpages, 1))

        converged = False
        for i in range(self.max_iter):
            new_pagerank = transition_matrix.mm(pagerank)
            new_pagerank = n_webpages * new_pagerank / ones.T.mm(new_pagerank)

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
