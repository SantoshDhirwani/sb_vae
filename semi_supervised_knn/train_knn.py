import argparse
import os
from os.path import join as pjoin
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from model import PseudoLabeler
from load_mnist import load_mnist, mkdirs

def build_argparser():
    DESCRIPTION = ("Train a Semi-Supervised KNN classifier.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    dataset = p.add_argument_group("Experiment options")
    dataset.add_argument('--dataset', default="mnist", choices=["mnist", "mnist_plus_rot", "svhn_pca"],
                         help="either 'mnist' or 'mnist_plus_rot' or 'svhn_pca'. Default:%(default)s")
    dataset.add_argument('--label-drop-percentage', type=float, default=.99,
                         help='percentage of labels to drop from training data. Default:%(default)s')

    model = p.add_argument_group("Model options")
    model.add_argument('--k', type=int, default=5,
                         help='number of nearest neighbors. Default:%(default)s')

    training = p.add_argument_group("Training options")
    training.add_argument('--sampling_rate', type=float, default=0,
                          help='percent of samples used as pseudo-labelled data from the unlabled dataset Default: %(default)s.')

    general = p.add_argument_group("General arguments")
    general.add_argument('--experiment-dir', default="./experiments/",
                         help='name of the folder where to save the experiment. Default: %(default)s.')
    return p

# get command-line args
parser = build_argparser()
args = parser.parse_args()
args_dict = vars(args)
args_string = ''.join('{}_{}_'.format(key, val) for key, val in sorted(args_dict.items()) if key not in ['experiment_dir'])[:-1]

# Create datasets and experiments folders is needed.
dataset_dir = mkdirs("./datasets")
mkdirs(args.experiment_dir)

if args.dataset == 'mnist':
    dataset = pjoin(dataset_dir, args.dataset + ".pkl")

print("Datasets dir: {}".format(os.path.abspath(dataset_dir)))
print("Experiment dir: {}".format(os.path.abspath(args.experiment_dir)))

if "mnist" in dataset:
    # We follow the approach used in [2] to split the MNIST dataset.
    datasets = load_mnist(dataset, target_as_one_hot=False, flatten=True, split=(45000, 5000, 10000), drop_percentage=args.label_drop_percentage)

inputs,labels = datasets

unlabeleld_data = inputs[2]
labeled_data = inputs[0]
y_labeled = labels[0]
y_unlabeled = labels[2]

train_x = pd.DataFrame(labeled_data)
test_x = pd.DataFrame(unlabeleld_data)
train_y  = pd.DataFrame(y_labeled, columns = ['labels'])
test_y = pd.DataFrame(y_unlabeled, columns = ['labels'])

features = train_x.columns
target = 'labels'

print('Creating Pseudo labels for model training.')
model = PseudoLabeler(
        KNeighborsClassifier(n_neighbors=args.k),
        test_x,
        features,
        target,
        sample_rate = args.sampling_rate
        )
print('Training the KNN model...')
model.fit(train_x, train_y)
pred = model.predict(test_x)
true_labels = test_y.values.flatten()
acc = accuracy_score(true_labels,pred)
Test_error  = np.subtract(1.0,acc)
acc = 100*round(acc,2)
Test_error = 100 * round(Test_error,2)
print(' Check Results...')
print(f'Accuracy : {acc} %, Test error : {Test_error} %')
