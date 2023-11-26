import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch


def plot_gradient_one_var():
    # Define the function f(x) = x^2
    def f(x):
        return x**2

    # Define the derivative of f(x)
    def df(x):
        return 2 * x

    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(-16, 16, 0.1)
    ax.plot(x, f(x), label=r"$f(x) = x^2$", c="b", linewidth=2)

    # Annotate points to show gradient direction
    points = [-6, -8, -10, -12, -14, 6, 8, 10, 12, 14]
    p = 0.05
    tilt_by = 0.3
    for x in points:
        y = f(x)
        arrow_tail = (x, y)
        gradient = df(x)
        dx = gradient * p
        x_ = x + dx  # changing x in the direction of gradient by 5%.
        # dy = f(x + dx) - f(x) # change in y wrt change in x in the direction of gradient by 5%.
        # y_ = y + dy
        y_ = f(x_)  # new y with for the new x
        arrow_head = (x_ + tilt_by, y_)

        if x != 0:
            arr = patch.FancyArrowPatch(
                arrow_tail,  # arrow tail
                arrow_head,  # arrow head
                arrowstyle="->,head_width=.15",
                mutation_scale=20,
                linewidth=3,
                color="r",
            )
            ax.add_patch(arr)
            # relative label
            # https://matplotlib.org/stable/users/explain/text/annotations.html#annotating-an-artist
            ax.annotate(
                r"$\nabla = $" + f"{gradient}",
                (0.5, 1),
                xycoords=arr,  ## (1,1) relative to arr
                ha="left",
                va="bottom",
                fontsize=12,
            )

    ax.set_xlabel("x")
    ax.set_ylabel(r"$f(x)$")
    ax.legend()
    ax.grid()
    ax.set_title(r"Function $f(x) = x^2$ and its gradient $\nabla f(x) = 2x$")
    plt.show()


def plot_gradient_descent_one_var():
    # Define the function f(x) = x^2
    def L(w):
        return w**2

    # Define the derivative of f(x)
    def dL(w):
        return 2 * w

    def gradient_descent(w, lr=0.045):
        gradient = dL(w)
        # change in w in the `opposite` direction of gradient by 5%.
        dw = -lr * gradient
        new_w = w + dw
        return new_w, dw, gradient

    fig, ax = plt.subplots(figsize=(10, 8))

    w = np.arange(-16, 16, 0.1)
    ax.plot(w, L(w), label=r"$L(w) = w^2$", c="b", linewidth=2)

    # Annotate points to show gradient direction
    points = [-6, -8, -10, -12, -14, 6, 8, 10, 12, 14]
    tilt_by = 0.3
    for w in points:
        loss = L(w)
        arrow_tail = (w, loss)
        new_w, dw, gradient = gradient_descent(w)
        # dLoss = L(w + dw) - L(w)
        # new_loss = (loss - dLoss)
        new_loss = L(new_w)
        arrow_head = (new_w + tilt_by, new_loss)

        if w != 0:
            arr = patch.FancyArrowPatch(
                arrow_tail,  # arrow tail
                arrow_head,  # arrow head
                arrowstyle="->,head_width=.15",
                mutation_scale=20,
                linewidth=3,
                color="r",
            )
            ax.add_patch(arr)
            # relative label
            # https://matplotlib.org/stable/users/explain/text/annotations.html#annotating-an-artist
            ax.annotate(
                r"$\nabla= $"
                + f"{gradient}, "
                + "\n"
                + r"$\delta w= $"
                + f"{round(-dw,2)}",
                (1, -0.2),
                xycoords=arr,  ## (1,1) relative to arr
                ha="left",
                va="bottom",
                fontsize=12,
            )

    ax.set_xlabel("w")
    ax.set_ylabel("L(w)")
    ax.legend()
    ax.grid()
    ax.set_title(r"Function $L(w) = w^2$ and its gradient $\nabla L(w) = 2w$")
    plt.show()


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def generate_blob_cluster(split_train_test=False):
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
