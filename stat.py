#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

from scipy.sparse import load_npz
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

    J1 = p_js1.size - 1
    J2 = p_js2.size - 1
    Sigma1 = np.count_nonzero(sigmas1) + 1
    Sigma2 = np.count_nonzero(sigmas2) + 1

    print('decision points:', J1, '+', J2, '=', J1 + J2)
    print('sequences:', Sigma1, '+', Sigma2, '=', Sigma1 + Sigma2)
    print('payoffs:', us.nnz)


if __name__ == '__main__':
    main()
