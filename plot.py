import numpy as np
import matplotlib.pyplot as plt


def plot_decision_regions(
    X, y, classifier, figsize=(8, 6), resolution=0.01, cmap=plt.cm.cool
):
    """
    Plot decision regions of a classifier.
    `X`: feature matrix
    `y`: target vector
    `classifier`: trained classifier
    `figsize`: figure size
    `resolution`: resolution of the meshgrid
    `cmap`: color map

    Example usage: `plot_decision_regions(X_train, y_train, classifier)`
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step=resolution),
        np.arange(y_min, y_max, step=resolution),
    )
    Z = classifier.predict(
        np.c_[xx.ravel(), yy.ravel()]  # ~ np.array([xx.ravel(), yy.ravel()]).T
    )
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors="k", marker="o", s=50)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
