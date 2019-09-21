import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


import numpy as np


# adding libraries directory to path for script to work both if called directly or imported
import sys
sketch_features_path = os.path.join(HERE_PATH, '..')
sys.path.append(sketch_features_path)

from sketch_features import tools


def scale_drawing(X):
    ## center and scale drawing to fit within [-1, 1] in both x and y dimension
    means = np.mean(X, axis=0)
    centered_X = np.array(X) - means

    maxs = np.max(centered_X, axis=0)
    mins = np.min(centered_X, axis=0)
    strecth_ratio = 1 / np.max(np.abs(np.concatenate((maxs, mins))))

    scaled_drawing = centered_X * strecth_ratio

    return scaled_drawing, strecth_ratio


def compute_drawing_descriptors(scaled_drawing):
    descriptors = []
    # add starting position
    descriptors.append(scaled_drawing[0,0])
    descriptors.append(scaled_drawing[0,1])

    # add displacement (end - start) vector and
    delta = scaled_drawing[-1,:] - scaled_drawing[0,:]
    descriptors.append(delta[0])
    descriptors.append(delta[1])

    # add crow flies displacement
    descriptors.append(np.linalg.norm(delta))

    # add cumulative displacement (lenght of the drawing)
    diff = np.diff(scaled_drawing, axis=0)
    drawing_lenght = np.sum(np.linalg.norm(diff, axis=1))
    descriptors.append(drawing_lenght)

    return descriptors


def compute_drawing_histogram_flatten(scaled_drawing, bins=3, bins_range=[[-1,1], [-1,1]]):
    x = scaled_drawing[:, 0]
    y = scaled_drawing[:, 1]
    H, _, _ = np.histogram2d(x, y, bins=bins, range=bins_range, density=True)
    return H.T.flatten().tolist()


def compute_embeddings_from_drawing(drawing):
    scaled_drawing, strecth_ratio = scale_drawing(drawing)
    descriptors = compute_drawing_descriptors(scaled_drawing)
    histogram = compute_drawing_histogram_flatten(scaled_drawing)

    embeddings = []
    embeddings.append(strecth_ratio / 10) # divide by 10 to make this feature about the same scale as the others
    embeddings.extend(descriptors)
    embeddings.extend(histogram)

    return embeddings


def compute_embeddings_from_file(drawing_file):
    drawing = tools.read_json(drawing_file)
    return compute_embeddings_from_drawing(drawing)
