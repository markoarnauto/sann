import numpy as np
from collections import deque


class LSH:
    '''
    This hashing schema is based on random projections.
    This code is based on this tutorial: https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23
    Some adaptions are made to support constant space and time hashing.
    '''
    def __init__(self, num_tables: int, hash_size: int, inp_dimension: int, bucket_size: int):
        self._num_tables = num_tables  # L
        self._hash_tables = list()
        for i in range(self._num_tables):
            self._hash_tables.append(HashTable(hash_size, inp_dimension, bucket_size))

    def __setitem__(self, vec: np.ndarray, id: int):
        for table in self._hash_tables:
            table[vec] = id

    def __getitem__(self, vec: np.ndarray) -> list:
        results_count = dict()
        for table in self._hash_tables:
            for coll_id in table[vec]:
                results_count[coll_id] = results_count.get(coll_id, 0) + 1
        # get ids with most collisions
        # return a maximum of 3L results
        return sorted(results_count, key=results_count.get)[:3 * self._num_tables]



class HashTable:
    def __init__(self, hash_size: int, inp_dimensions: int, bucket_size):
        self._bucket_size = bucket_size  # for constant space approach
        self._hash_size = hash_size
        self._inp_dimensions = inp_dimensions
        self._hash_table = dict()
        self._projections = np.random.randn(self._hash_size, inp_dimensions)

    def _generate_hash(self, vec: np.ndarray):
        bools = (np.dot(vec, self._projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, vec: np.ndarray, id: int):
        hash_value = self._generate_hash(vec)
        # if bucket does not exist create one (a deque with fixed lenght -> constant space approach)
        self._hash_table.setdefault(hash_value, deque(maxlen=self._bucket_size))
        self._hash_table[hash_value].appendleft(id)

    def __getitem__(self, vec: np.ndarray) -> list:
        hash_value = self._generate_hash(vec)
        return list(self._hash_table.get(hash_value, []))

