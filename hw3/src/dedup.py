"""Exact Jaccard search over word shingles, including all pairs above threshold.

A prefix index narrows candidates without probabilistic MinHash/LSH omissions.
The common token order and prefix length n-ceil(threshold*n)+1 guarantee that
qualifying pairs share an indexed prefix token. Every candidate is then checked
with the actual Jaccard coefficient. num_perm stays in the public interface for
compatibility with the course scaffold; this exact implementation does not use it.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from src.textnorm import shingles


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            lo, hi = sorted((a, b))
            self.parent[hi] = lo


def exact_duplicates(keys: Sequence[str]) -> list[int]:
    seen, duplicates = set(), []
    for i, key in enumerate(keys):
        if key in seen:
            duplicates.append(i)
        else:
            seen.add(key)
    return duplicates


def _pairs(left, right, threshold, within):
    if not 0 < threshold <= 1:
        raise ValueError("Jaccard threshold must be in (0, 1]")
    vocabulary = Counter(
        token for group in (left if within else [*left, *right]) for token in group
    )

    def prefix(tokens):
        order = sorted(tokens, key=lambda token: (vocabulary[token], token))
        count = len(tokens) - math.ceil(threshold * len(tokens) - 1e-12) + 1
        return order[: max(count, 0)]

    postings = defaultdict(list)
    empty = []

    def insert(index, tokens):
        if not tokens:
            empty.append(index)
        for token in prefix(tokens):
            postings[token].append(index)

    if not within:
        for i, tokens in enumerate(left):
            insert(i, tokens)
    for j, tokens in enumerate(right):
        candidates = set()
        for token in prefix(tokens):
            candidates.update(postings[token])
        if not tokens:
            candidates.update(empty)
        for i in sorted(candidates):
            other = left[i]
            if min(len(tokens), len(other)) + 1e-12 < threshold * max(len(tokens), len(other)):
                continue
            intersection = len(tokens & other)
            union = len(tokens) + len(other) - intersection
            score = intersection / union if union else 1.0
            if score + 1e-12 >= threshold:
                yield i, j
        if within:
            insert(j, tokens)


def similar_set_pairs(token_sets, threshold):
    return list(_pairs(token_sets, token_sets, threshold, within=True))


def near_pairs(texts, shingle_words, threshold):
    return similar_set_pairs([shingles(text, shingle_words) for text in texts], threshold)


def near_duplicates(texts, shingle_words, num_perm, threshold):
    groups = UnionFind(range(len(texts)))
    for i, j in near_pairs(texts, shingle_words, threshold):
        groups.union(i, j)
    return [i for i in range(len(texts)) if groups.find(i) != i]


def cross_near_duplicates(left, right, shingle_words, num_perm, threshold):
    lsets = [shingles(text, shingle_words) for text in left]
    rsets = [shingles(text, shingle_words) for text in right]
    return list(_pairs(lsets, rsets, threshold, within=False))
