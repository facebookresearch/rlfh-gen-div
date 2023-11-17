"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


def map(f, n):
    if isinstance(n, tuple) or isinstance(n, list):
        return n.__class__(map(f, sn) for sn in n)
    elif isinstance(n, dict):
        return {k: map(f, v) for k, v in n.items()}
    else:
        return f(n)


def dict_map(f, n):
    if isinstance(n, dict):
        return {k: dict_map(f, v) for k, v in n.items()}
    else:
        return f(n)


def flatten(n):
    if isinstance(n, tuple) or isinstance(n, list):
        for sn in n:
            yield from flatten(sn)
    elif isinstance(n, dict):
        for key in n:
            yield from flatten(n[key])
    else:
        yield n


def index(n, indices):
    if isinstance(n, tuple) or isinstance(n, list):
        return type(n)([n[i] for i in indices])
    elif isinstance(n, dict):
        return {k: index(v, indices) for k, v in n.items()}
    else:
        return n[indices]


def zip(*nests):
    n0, *nests = nests
    iters = [flatten(n) for n in nests]

    def f(first):
        return [first] + [next(i) for i in iters]

    return map(f, n0)


def map_many(f, *nests):
    n0, *nests = nests
    iters = [flatten(n) for n in nests]

    def g(first):
        return f([first] + [next(i) for i in iters])

    return map(g, n0)


def find_nested(filter, d):
    """Find all nested values in a dict that match a filter"""
    if isinstance(d, dict):
        for k, v in d.items():
            if filter(k):
                yield v
            yield from find_nested(filter, v)
    elif isinstance(d, list):
        for v in d:
            yield from find_nested(filter, v)
    elif isinstance(d, tuple):
        for v in d:
            yield from find_nested(filter, v)


def remove_nested(filter, d):
    """Return d but with all nested keys that match a filter removed"""
    if isinstance(d, dict):
        return {k: remove_nested(filter, v) for k, v in d.items() if not filter(k)}
    else:
        return d
