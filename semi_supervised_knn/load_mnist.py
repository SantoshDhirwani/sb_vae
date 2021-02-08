import numpy as np
import random
import _pickle as cPickle
import gzip
import zipfile
import tarfile
import urllib.request
import os

def _split_data(data, split):
    starts = np.cumsum(np.r_[0, split[:-1]])
    ends = np.cumsum(split)
    splits = [data[s:e] for s, e in zip(starts, ends)]
    return splits
# Load MNIST data
def load_mnist(path, target_as_one_hot=False, flatten=False, split=(50000, 10000, 10000), drop_percentage=None):
    ''' Loads the MNIST dataset.
    Input examples are 28x28 pixels grayscaled images. Each input example is represented
    as a ndarray of shape (28*28), i.e. (height*width).
    Example labels are integers between [0,9] respresenting one of the ten classes.
    Parameters
    ----------
    path : str
        The path to the dataset file (.npz).
    target_as_one_hot : {True, False}, optional
        If True, represent targets as one hot vectors.
    flatten : {True, False}, optional
        If True, represents each individual example as a vector.
    split : tuple of int, optional
        Numbers of examples in each split of the dataset. Default: (50000, 10000, 10000)
    References
    ----------
    This dataset comes from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/
    '''
    if not os.path.isfile(path):
        # Download the dataset.
        data_dir, data_file = os.path.split(path)
        mnist_picklefile = os.path.join(data_dir, 'mnist.pkl.gz')

        if not os.path.isfile(mnist_picklefile):
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print("Downloading data (16 Mb) from {} ...".format(origin))
            urllib.request.urlretrieve(origin, mnist_picklefile)

        # Load the dataset and process it.
        inputs = []
        labels = []
        print("Processing data ...")
        with gzip.open(mnist_picklefile, 'rb') as f:
            trainset, validset, testset = cPickle.load(f,encoding='latin1')

        inputs = np.concatenate([trainset[0], validset[0], testset[0]], axis=0).reshape((-1, 1, 28, 28))
        labels = np.concatenate([trainset[1], validset[1], testset[1]], axis=0).astype(np.int8)


    if flatten:
        inputs = inputs.reshape((len(inputs), -1))

    #shuffle
    idxs = list(range(inputs.shape[0]))
    random.shuffle(idxs)
    inputs = inputs[idxs,:]
    labels = labels[idxs]

    if target_as_one_hot:
        one_hot_vectors = np.zeros((labels.shape[0], 10), dtype=theano.config.floatX)
        one_hot_vectors[np.arange(labels.shape[0]), labels] = 1
        labels = one_hot_vectors

    datasets_inputs = _split_data(inputs, split)
    datasets_labels = _split_data(labels, split)

    if drop_percentage > 0.:
        N_train = split[0]
        N_wo_label = int(drop_percentage * N_train)
        # split inputs
        labeled_data = datasets_inputs[0][N_wo_label:,:]
        unlabeled_data = datasets_inputs[0][:N_wo_label,:]
        datasets_inputs[0] = labeled_data
        datasets_inputs.insert(2, unlabeled_data)
        # split labels
        labeled_data = datasets_labels[0][N_wo_label:]
        unlabeled_data = datasets_labels[0][:N_wo_label]
        datasets_labels[0] = labeled_data
        datasets_labels.insert(2, unlabeled_data)

    return datasets_inputs, datasets_labels

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass
    return path
