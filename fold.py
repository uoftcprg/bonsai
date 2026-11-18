#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from scipy.sparse import coo_array, load_npz, save_npz
import numpy as np


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('sfps', type=Path)
    parser.add_argument('us', type=Path)
    parser.add_argument('sfps2', type=Path)
    parser.add_argument('us2', type=Path)
    parser.add_argument('maps', nargs='?', type=Path)
    parser.add_argument('--dtype', default='double')
    parser.add_argument('--itype', default='intc')

    return parser.parse_args()


def number(colors):
    _, heads, indices = np.unique(
        colors,
        return_index=True,
        return_inverse=True,
    )

    return np.unique(heads[indices], return_inverse=True)


def fold(p_js, sigmas, colors_j, colors_sigma):
    heads_j, classes_j = number(colors_j)
    heads_sigma, classes_sigma = number(colors_sigma)
    folded_p_js = classes_sigma[p_js[heads_j]]
    folded_sigmas = classes_j[sigmas[heads_sigma]]
    _, indices, counts = np.unique(
        np.stack((sigmas, classes_sigma)),
        axis=1,
        return_inverse=True,
        return_counts=True,
    )
    counts = counts[indices.ravel()]
    weights = np.ones(sigmas.size)
    ancestors = np.arange(sigmas.size)

    while ancestors.any():
        weights *= counts[ancestors]
        ancestors = p_js[sigmas[ancestors]]

    return folded_p_js, folded_sigmas, heads_sigma, classes_sigma, weights


def main():
    args = parse_args()
    colors = defaultdict(int)
    status = True

    while status:
        try:
            line = input()
        except EOFError:
            status = False
        else:
            tokens = line.split()

            match tokens:
                case 's', 'col', _:
                    pass
                case 'l', v, c:
                    v = int(v)
                    c = int(c)
                    colors[v] = c
                case _:
                    raise NotImplementedError

    p_js1, sigmas1, p_js2, sigmas2 = np.load(args.sfps).values()
    us = load_npz(args.us).astype(float)
    p_js1_offset = 1
    sigmas1_offset = p_js1_offset + p_js1.size
    p_js2_offset = sigmas1_offset + sigmas1.size
    sigmas2_offset = p_js2_offset + p_js2.size
    us_offset = sigmas2_offset + sigmas2.size
    colors = np.array([colors[v] for v in range(us_offset)])
    p_js1, sigmas1, heads1, classes1, weights1 = fold(
        p_js1,
        sigmas1,
        colors[p_js1_offset:sigmas1_offset],
        colors[sigmas1_offset:p_js2_offset],
    )
    p_js2, sigmas2, heads2, classes2, weights2 = fold(
        p_js2,
        sigmas2,
        colors[p_js2_offset:sigmas2_offset],
        colors[sigmas2_offset:us_offset],
    )
    indices = np.flatnonzero(np.isin(us.row, heads1))
    us = coo_array(
        (
            us.data[indices],
            (
                classes1[us.row[indices]].astype(args.itype),
                classes2[us.col[indices]].astype(args.itype),
            ),
        ),
        (sigmas1.size, sigmas2.size),
    )

    us.sum_duplicates()

    us.data *= np.bincount(classes1)[us.row]
    us.data /= weights1[heads1][us.row] * weights2[heads2][us.col]

    us.eliminate_zeros()

    args.sfps2.parent.mkdir(parents=True, exist_ok=True)
    args.us2.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.sfps2,
        p_js1.astype(args.itype),
        sigmas1.astype(args.itype),
        p_js2.astype(args.itype),
        sigmas2.astype(args.itype),
    )
    save_npz(args.us2, us.astype(args.dtype))

    if args.maps is not None:
        args.maps.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.maps,
            classes1.astype(args.itype),
            1 / weights1,
            classes2.astype(args.itype),
            1 / weights2,
        )


if __name__ == '__main__':
    main()
