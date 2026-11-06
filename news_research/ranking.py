"""Small-corpus BM25 and reciprocal-rank fusion without external services."""
import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_rank(texts, question, k):
    if not texts:
        return []
    terms = [Counter(tokenize(text)) for text in texts]
    lengths = [sum(row.values()) for row in terms]
    average = sum(lengths) / len(lengths) or 1
    scores = [0.0] * len(texts)
    for word in set(tokenize(question)):
        frequency = sum(word in row for row in terms)
        idf = math.log(1 + (len(texts) - frequency + .5) / (frequency + .5))
        for i, row in enumerate(terms):
            count = row[word]
            scores[i] += idf * count * 2.5 / (count + 1.5 * (.25 + .75 * lengths[i] / average))
    return [i for i in sorted(range(len(texts)), key=lambda i: (-scores[i], i)) if scores[i] > 0][:k]


def reciprocal_rank_fusion(rankings, k, constant=60):
    scores = {}
    for ranking in rankings:
        for position, ident in enumerate(dict.fromkeys(ranking), 1):
            scores[ident] = scores.get(ident, 0) + 1 / (constant + position)
    return sorted(scores, key=lambda ident: (-scores[ident], str(ident)))[:k]
