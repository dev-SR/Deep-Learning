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


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx : start_idx + minibatch_size]

        yield X[batch_idx], y[batch_idx]


def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


def mse_loss(targets, probas, num_labels=2):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas) ** 2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


def compute_mse_and_acc(model, X, y, num_labels=2, minibatch_size=100):
    mse, correct_pred, num_examples = 0.0, 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):
        probas = model(features)
        predicted_labels = np.argmax(probas, axis=1)

        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        correct_pred += (predicted_labels == targets).sum()

        num_examples += targets.shape[0]
        mse += loss

    mse = mse / i
    acc = correct_pred / num_examples
    return mse, acc
