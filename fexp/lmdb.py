# -*- coding: utf-8 -*-
"""Tools to create LMDB database and subsequently access it for efficient memory IO.
Based on code from Jonas Teeuwen:
https://github.com/deepmedic/manet/blob/master/examples/create_lmdb_set.py
https://github.com/deepmedic/manet/blob/master/manet/lmdb/dataset.py

"""
import lmdb
import os
import copy
from tqdm import tqdm
try:
    import simplejson as json
except ImportError:
    import json
import numpy as np
from .utils import read_list, write_list

def write_kv_to_lmdb(db, key, value):
    """
    Write (key, value) to db.
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key, value)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit * 2
            tqdm.write('MapFullError: Doubling LMDB map size to {}MB.'.format(new_limit))
            db.set_mapsize(new_limit)


def write_data_to_lmdb(db, key, image, metadata):
    """Write image data to db."""
    write_kv_to_lmdb(db, key, np.ascontiguousarray(image).tobytes())
    meta_key = key + '_metadata'
    ser_meta = json.dumps(metadata)
    write_kv_to_lmdb(db, meta_key, ser_meta)


def build_db(path, db_name, cases, load_fn):
    """Build LMDB with images from load_fn

    Parameters
    ----------
    path : str
        Path to folder with LMDB db.
    db_name : str
        Name of the database.
    cases : list of str
        Name of all cases to write to database (these become the database keys)
    load_fn : function which fetches the data associated with each element in cases.
        The function should return a list or iterable of ndarrays.

    """
    db = lmdb.open(os.path.join(path, db_name), map_async=True, max_dbs=0, writemap=True)

    for case in tqdm(cases):
        ndarrays = load_fn(case)
        listlen = len(ndarrays)
        key = "{}_len".format(case)
        write_kv_to_lmdb(db, key, json.dumps(listlen))
        for i, data in enumerate(ndarrays):
            metadata = dict(shape=data.shape, dtype=str(data.dtype))
            key = "{}_{}".format(case, i)
            write_data_to_lmdb(db, key, data, metadata)

    db.close()

    # write case keys to database key file
    lmdb_keys_path = os.path.join(path, db_name + '_keys.json')
    write_list(cases, lmdb_keys_path)


class LmdbDb(object):
    def __init__(self, path, db_name):
        """Load an LMDB database, containing a dataset.
        The dataset should be structured as image_id: binary representing the contiguous block.
        If image_id is available we also need image_id_metadata which is a json parseble dictionary.
        This dictionary should contains the key 'shape' representing the shape and 'dtype'.
        If the keys file is available, the file is loaded, otherwise generated.
        Parameters
        ----------
        path : str
            Path to folder with LMDB db.
        db_name : str
            Name of the database.
        """
        lmdb_path = os.path.join(path, db_name)
        lmdb_keys_path = os.path.join(path, db_name + '_keys.json')
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, max_readers=None, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        if os.path.isfile(lmdb_keys_path):
            self._keys = read_list(lmdb_keys_path)
        else:
            # The keys file does not exist, we will generate one, but this can take a while.
            with self.env.begin(write=False) as txn:
                keys = [key[:-len('_len')] for key, _ in txn.cursor() if '_len' in key]
                write_list(keys, lmdb_keys_path)
                self._keys = keys

    def __delitem__(self, key):
        idx = self._keys.index[key]
        self._keys.pop(idx, None)

    def copy(self):
        return copy.deepcopy(self)

    def has_key(self, key):
        return key in self._keys

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        with self.env.begin(buffers=True, write=False) as txn:
            if key not in self._keys:
                raise KeyError(key)
            itemlen = json.loads(str(txn.get(key + '_len')))
            result = []
            for i in range(itemlen):
                result.append(self._getsubitem(key + '_{}'.format(i), txn))
        return result

    def _getsubitem(self, key, txn):
        buf = txn.get(key)
        meta_buf = txn.get(key + '_metadata')

        metadata = json.loads(str(meta_buf))
        dtype = metadata['dtype']
        shape = metadata['shape']
        data = np.ndarray(shape, dtype, buffer=buf)

        return data #dict(data=data, metadata=metadata)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.lmdb_path + ')'

    def close(self):
        self.env.close()
        self.keys = None