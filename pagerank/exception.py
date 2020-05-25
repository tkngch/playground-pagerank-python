#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pagerank.exception."""


class ConvergenceError(RuntimeError):
    """The exception to raise when the algorithm failed to reach the convergence."""
