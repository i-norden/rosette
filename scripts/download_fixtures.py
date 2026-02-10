#!/usr/bin/env python3
"""Download test fixtures. Thin wrapper around snoopy.demo.fixtures."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snoopy.demo.fixtures import download_all

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_all()
