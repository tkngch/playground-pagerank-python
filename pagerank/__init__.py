#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pagerank."""

from .pagerank_with_numpy import PageRankNumpy
from .pagerank_with_torch import PageRankTorch

__all__ = ["PageRankNumpy", "PageRankTorch"]
