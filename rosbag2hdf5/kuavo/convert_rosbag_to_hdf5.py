#!/usr/bin/env python3
"""Semantic CLI alias for rosbag->HDF5 conversion."""

from converter.common.logging_utils import setup_logging


if __name__ == "__main__":
    setup_logging()
    from converter.pipeline.conversion_orchestrator import main

    main()
