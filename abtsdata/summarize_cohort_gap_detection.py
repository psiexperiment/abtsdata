from pathlib import Path

from . import _summarize_behavior as behavior


def main_gonogo():
    import argparse
    parser = argparse.ArgumentParser('summarize-cohort-gap-detection')
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    behavior.summarize_cohort('gap_detection', 'gap', 0, path=args.path)
