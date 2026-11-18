#!/usr/bin/env python3

from argparse import ArgumentParser
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


def collapse(p_js, sigmas):
    counts = np.bincount(sigmas, minlength=p_js.size)
    counts[0] = 0
    targets = np.arange(sigmas.size)
    indices = np.flatnonzero(counts[sigmas] == 1)
    targets[indices] = p_js[sigmas[indices]]

    while np.any(targets != targets[targets]):
        targets = targets[targets]

    return targets


def trim(p_js, sigmas, targets, indices):
    live = np.zeros(sigmas.size, bool)
    live[indices] = True
    live[0] = True
    parents = targets[p_js[sigmas]]
    count = 0

    while count != np.count_nonzero(live):
        count = np.count_nonzero(live)
        live[parents[live]] = True

    mask_j = np.zeros(p_js.size, bool)
    mask_j[sigmas[live]] = True
    mask_sigma = mask_j[sigmas] & (sigmas != 0)
    mask_sigma[0] = True
    indices_j = np.cumsum(mask_j) - 1
    indices_sigma = np.cumsum(mask_sigma) - 1
    trimmed_p_js = indices_sigma[targets[p_js[mask_j]]]
    trimmed_sigmas = indices_j[sigmas[mask_sigma]]
    firsts = np.zeros(sigmas.size, bool)
    firsts[np.unique(sigmas, return_index=True)[1]] = True
    indices = np.arange(sigmas.size)
    scales = np.ones(sigmas.size)

    while not np.all(mask_sigma[indices]):
        mask = ~mask_sigma[indices]
        scales[mask] *= firsts[indices[mask]]
        indices[mask] = p_js[sigmas[indices[mask]]]

    return trimmed_p_js, trimmed_sigmas, indices_sigma[indices], scales


def main():
    args = parse_args()
    p_js1, sigmas1, p_js2, sigmas2 = np.load(args.sfps).values()
    us = load_npz(args.us).astype(args.dtype)

    us.sum_duplicates()
    us.eliminate_zeros()

    targets1 = collapse(p_js1, sigmas1)
    targets2 = collapse(p_js2, sigmas2)
    us = coo_array((us.data, (targets1[us.row], targets2[us.col])), us.shape)

    us.sum_duplicates()
    us.eliminate_zeros()

    p_js1, sigmas1, indices1, scales1 = trim(p_js1, sigmas1, targets1, us.row)
    p_js2, sigmas2, indices2, scales2 = trim(p_js2, sigmas2, targets2, us.col)
    us = coo_array(
        (
            us.data,
            (
                indices1[us.row].astype(args.itype),
                indices2[us.col].astype(args.itype),
            ),
        ),
        (sigmas1.size, sigmas2.size),
    )

    args.sfps2.parent.mkdir(parents=True, exist_ok=True)
    args.us2.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.sfps2,
        p_js1.astype(args.itype),
        sigmas1.astype(args.itype),
        p_js2.astype(args.itype),
        sigmas2.astype(args.itype),
    )
    save_npz(args.us2, us)

    if args.maps is not None:
        args.maps.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.maps,
            indices1.astype(args.itype),
            scales1,
            indices2.astype(args.itype),
            scales2,
        )


if __name__ == '__main__':
    main()
