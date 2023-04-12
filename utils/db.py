import json
import os
from docarray import DocumentArray, Document
import numpy as np
from annlite import AnnLite


def write_samples(keys, embs):

    da = DocumentArray()
    for k, emb in zip(keys, embs):
        da.append(Document(id=k, embedding=emb))

    da.save('data.json', file_format='json')


def query_sample(key, emb):
    da = DocumentArray.load('data.json', file_format='json')
    q = Document(id=key, embedding=emb).match(da, metric='cosine')
    return [{'id': m.id, 'scores': m.scores}
            for m in q.matches]


def write_sample(key, emb, meta):

    output_dir = 'DB'
    os.makedirs(output_dir, exist_ok=True)

    data = {
        'key': key,
        'embedding': emb,
        'metadata': meta
    }

    with open(f'{output_dir}/{key}.json', 'w') as f:
        json.dump(data, f)
