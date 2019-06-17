import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

from collections import deque

import umap
import numpy as np

# adding libraries directory to path for script to work both if called directly or imported
import sys
sketch_features_path = os.path.join(HERE_PATH, '..')
sys.path.append(sketch_features_path)

from sketch_features import embedding_tools


class SketchVaultSignal(object):

    def __init__(self):
        self.embeddings = []

    def add_feedback_sketch(self, feedback_sketch_file):
        embedding = embedding_tools.compute_embeddings_from_file(feedback_sketch_file)
        self.embeddings.append(embedding)

    def get_feedback_signals(self):

        X_umap = np.array(self.embeddings)
        
        if X_umap.shape[0] < 4:
            # not enough point yet for umap to compute anything
            # we are just returning two first column of the embeddings
            # should not be used by the algorithm down the pipe at this point
            X_mapped = X_umap[:, :2]
            
        else:
            n_neighbors = min(15, X_umap.shape[0] - 1)

            X_mapped = umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=2,
                                 min_dist=0.5,
                                 metric='correlation').fit_transform(X_umap)

        ##
        results = {}
        results['embeddings_for_umap'] = X_umap
        results['mapped_embeddings_from_umap'] = X_mapped

        return X_mapped.tolist(), results

    

class SketchVaultPlayer(object):

    def __init__(self, n_hypothesis, positive_sketch_file_list, negative_sketch_file_list, target_index=None):
        self.n_hypothesis = n_hypothesis
        self.positive_sketch_file_deque = deque(positive_sketch_file_list)
        self.negative_sketch_file_deque = deque(negative_sketch_file_list)
        self.update_target_index(target_index)

    def update_target_index(self, new_target_index=None):
        if new_target_index is None:  # set it randomly
            self.target_index = np.random.randint(self.n_hypothesis)
        else:
            self.target_index = new_target_index

    def get_sketch_file(self, flash_pattern):
        is_target_flashed = flash_pattern[self.target_index]
        if is_target_flashed:
            deque_to_sample = self.positive_sketch_file_deque
        else:
            deque_to_sample = self.negative_sketch_file_deque

        deque_to_sample.rotate()
        return deque_to_sample[0]
