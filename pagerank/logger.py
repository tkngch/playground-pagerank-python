#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pagerank.logger."""

import logging
import sys


def get_logger(name: str, level: str) -> logging.Logger:
    """Get an instance of Logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
