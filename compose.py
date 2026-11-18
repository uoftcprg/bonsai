#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('maps', type=Path)
    parser.add_argument('maps2', type=Path)
    parser.add_argument('maps3', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    indices1, scales1, indices2, scales2 = np.load(args.maps).values()
    indices3, scales3, indices4, scales4 = np.load(args.maps2).values()

    np.savez_compressed(
        args.maps3,
        indices3[indices1],
        scales1 * scales3[indices1],
        indices4[indices2],
        scales2 * scales4[indices2],
    )


if __name__ == '__main__':
    main()
