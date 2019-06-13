import os
import fnmatch

import json

import random
import numpy as np

def set_seed(seed, verbose=True):
    if verbose:
        print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)


def read_json(json_filename):
    with open(json_filename) as f:
        data = json.load(f)
    return data


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def list_files(path='.', patterns=['*'], min_depth=0, max_depth=float('inf')):
    if type(patterns) == str:
        patterns = [patterns]
    found_files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            filedir = os.path.abspath(dirpath)
            filepath = os.path.join(filedir, filename)
            depth = filepath[len(os.path.abspath(path)) + len(os.path.sep):].count(os.path.sep)
            if min_depth <= depth <= max_depth:
                for pattern in patterns:
                    if fnmatch.fnmatch(filename, pattern):
                        found_files.append(filepath)
    return found_files


def list_folders(path='.'):
    paths = [os.path.abspath(os.path.normpath(os.path.join(path, x))) for x in os.listdir(path)]
    return [path for path in paths if os.path.isdir(path)]
