#!/usr/bin/env python3

from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--dtype', default='ulonglong')
    parser.add_argument('--itype', default='intc')
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    header = input().split()

    if header[:2] != ['p', 'edge']:
        raise ValueError('not DIMACS graph')

    V, E = map(int, header[2:])
    vertices = {}
    colors = {}
    vertex_colors = np.zeros(V, args.dtype)
    edges = []

    with tqdm(total=E) as pbar:
        status = True

        while status:
            try:
                line = input()
            except EOFError:
                status = False
            else:
                tokens = line.split()

                match tokens:
                    case 'n', v, c:
                        v = vertices.setdefault(v, len(vertices))
                        c = colors.setdefault(c, len(colors) + 1)

                        if vertex_colors[v]:
                            raise ValueError('bad color')

                        vertex_colors[v] = c
                    case 'e', u, v:
                        u = vertices.setdefault(u, len(vertices))
                        v = vertices.setdefault(v, len(vertices))

                        if u > v:
                            u, v = v, u

                        edges.append((u, v))
                        pbar.update(1)
                    case _:
                        raise NotImplementedError

    if len(edges) != E:
        raise ValueError('bad number of edges')
    elif len(edges) != len(set(edges)):
        raise ValueError('duplicate edges')

    edges = np.array(edges, args.itype)
    _, vertex_colors = np.unique(vertex_colors, return_inverse=True)
    vertex_colors = vertex_colors.astype(args.dtype)
    rng = np.random.default_rng(args.seed)
    count = vertex_colors.max() + 1
    status = True

    while status:
        hashes = rng.integers(
            np.iinfo(args.dtype).max,
            size=count,
            dtype=args.dtype,
            endpoint=True,
        )
        sums = np.zeros(V, args.dtype)

        np.add.at(sums, edges[:, 0], hashes[vertex_colors[edges[:, 1]]])
        np.add.at(sums, edges[:, 1], hashes[vertex_colors[edges[:, 0]]])

        order = np.lexsort((sums, vertex_colors))
        splits = np.diff(vertex_colors[order]) != 0
        splits |= np.diff(sums[order]) != 0
        vertex_colors = np.empty(V, args.dtype)
        vertex_colors[order] = np.concatenate(([0], np.cumsum(splits)))
        previous_count = count
        count = vertex_colors.max() + 1
        status = previous_count != count

    print('s', 'col', count)

    for v, i in vertices.items():
        print('l', v, vertex_colors[i] + 1)


if __name__ == '__main__':
    main()
