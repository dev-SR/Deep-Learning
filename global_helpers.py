import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def generate_blob_cluster(split_train_test=False):
    """
    Usage:
        X,y = generate_blob_cluster()

        X_train, X_test, y_train, y_test = generate_blob_cluster(split_train_test=True)
    """
    NUM_FEATURES = 2
    NUM_CLASSES = 2
    RANDOM_SEED = 42
    X, y = make_blobs(
        n_samples=1000,
        n_features=NUM_FEATURES,
        centers=NUM_CLASSES,
        random_state=RANDOM_SEED,
        cluster_std=2.5,
    )
    print(X.shape, y.shape)

    # # Convert to tensors
    # data = torch.from_numpy(data).type(torch.float)
    # labels = torch.from_numpy(labels).type(torch.LongTensor)
    # mymap = matplotlib.colors.LinearSegmentedColormap.from_list("",['red','yellow','green','blue'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.cool, edgecolors="k", s=50)
    plt.show()
    if not split_train_test:
        return X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=RANDOM_SEED
        )
        return X_train, X_test, y_train, y_test
