#!/usr/bin/env python3
"""Download test fixtures. Thin wrapper around snoopy.demo.fixtures.

Usage: python -m scripts.download_fixtures   (from project root with snoopy installed)
"""

import logging

from snoopy.demo.fixtures import download_all

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_all()
