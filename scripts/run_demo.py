#!/usr/bin/env python3
"""Run snoopy demo. Thin wrapper around snoopy.demo.runner.

Usage: python -m scripts.run_demo   (from project root with snoopy installed)
"""

import argparse
import logging

from snoopy.demo.runner import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run snoopy demo with test fixtures")
    parser.add_argument("--download-only", action="store_true", help="Only download fixtures")
    parser.add_argument("--skip-llm", action="store_true", default=True, help="Skip LLM analysis")
    parser.add_argument("--output-dir", default=None, help="Output directory for HTML reports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_demo(download_only=args.download_only, skip_llm=args.skip_llm, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
