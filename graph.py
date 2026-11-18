#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

from scipy.sparse import load_npz
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('sfps', type=Path)
    parser.add_argument('us', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    p_js1, sigmas1, p_js2, sigmas2 = np.load(args.sfps).values()
    us = load_npz(args.us)
    p_js1_offset = 1
    sigmas1_offset = p_js1_offset + p_js1.size
    p_js2_offset = sigmas1_offset + sigmas1.size
    sigmas2_offset = p_js2_offset + p_js2.size
    us_offset = sigmas2_offset + sigmas2.size
    V = us_offset + us.nnz - 1
    E = (
        us_offset
        + 2 * us.nnz
        - (sigmas1.size - np.count_nonzero(sigmas1))
        - (sigmas2.size - np.count_nonzero(sigmas2))
        - 1
    )

    print('p', 'edge', V, E)

    for j, p_j in enumerate(tqdm(p_js1)):
        j += p_js1_offset
        p_j += sigmas1_offset

        print('n', j, 1)
        print('e', j, p_j)

    for sigma, j in enumerate(tqdm(sigmas1)):
        if j:
            sigma += sigmas1_offset
            j += p_js1_offset

            print('n', sigma, 2)
            print('e', sigma, j)

    for j, p_j in enumerate(tqdm(p_js2)):
        j += p_js2_offset
        p_j += sigmas2_offset

        print('n', j, 3)
        print('e', j, p_j)

    for sigma, j in enumerate(tqdm(sigmas2)):
        if j:
            sigma += sigmas2_offset
            j += p_js2_offset

            print('n', sigma, 4)
            print('e', sigma, j)

    _, data = np.unique(us.data, return_inverse=True)

    for u, (d, r, c) in enumerate(zip(tqdm(data), us.row, us.col)):
        u += us_offset
        r += sigmas1_offset
        c += sigmas2_offset

        print('n', u, d + 5)
        print('e', u, r)
        print('e', u, c)


if __name__ == '__main__':
    main()
