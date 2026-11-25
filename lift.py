#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('maps', type=Path)

    return parser.parse_args()


def main():
    args = parse_args()
    maps = np.load(args.maps).values()
    status = True

    while status:
        try:
            line = input()
        except EOFError:
            status = False
        else:
            match line.split():
                case 's', *x:
                    indices, scales, *maps = maps
                    x = scales * np.array(x, float)[indices]

                    print('s', *x.tolist())
                case _:
                    print(line)


if __name__ == '__main__':
    main()
