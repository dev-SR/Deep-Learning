# helpers
import tensorflow as tf
import tensorflow_hub as hub


import os
import random
from pathlib import Path
import shutil
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


def split_dataset_train_test(
    directory, train_dir, valid_dir, test_size=0.3, random_state=42
):
    """
    Split a dataset into train and validation sets and copy the files to the respective directories.

    Args:
        directory (str): Path to the directory containing the image dataset.
        train_dir (str): Path to the directory where the train set will be copied.
        valid_dir (str): Path to the directory where the validation set will be copied.
        test_size (float): The proportion of the dataset to include in the validation set (default: 0.3).
        random_state (int): Random seed for reproducibility (default: 42).

    Example of file structure of input:

    input <- top level folder
    └───pizza
    │   │   1008104.jpg
    │   │   1638227.jpg
    │   │   ...
    └───steak
        │   1000205.jpg
        │   1647351.jpg
        │   ...

    Example of file structure of output:
    output <- top level folder
    └───train <- training images
    │   └───pizza
    │   │   │   1008104.jpg
    │   │   │   1638227.jpg
    │   │   │   ...
    │   └───steak
    │       │   1000205.jpg
    │       │   1647351.jpg
    │       │   ...
    │
    └───test <- testing images
    │   └───pizza
    │   │   │   1001116.jpg
    │   │   │   1507019.jpg
    │   │   │   ...
    │   └───steak
    │       │   100274.jpg
    │       │   1653815.jpg
    │       │   ...
    """

    # Create train and valid directories if they don't exist
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(valid_dir).mkdir(parents=True, exist_ok=True)

    # Get all the image files in the directory
    root = Path(directory)
    all_files = list(root.glob("*/*"))

    # Split the files into train and valid sets
    train_files, valid_files = train_test_split(
        all_files, test_size=test_size, random_state=random_state
    )

    # Copy train files to train directory
    for file in tqdm(train_files, desc="Copying train files"):
        dest_path = str(file).replace(directory, train_dir)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(file), dest_path)

    # Copy valid files to valid directory
    for file in tqdm(valid_files, desc="Copying valid files"):
        dest_path = str(file).replace(directory, valid_dir)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(file), dest_path)

    # Print the number of files in the train and valid sets
    print("Number of train files:", len(train_files))
    print("Number of valid files:", len(valid_files))


def create_subset_dataset(dataset_name, percentage, new_path):
    """
    Creates a subset dataset by randomly selecting a percentage of images from the "train" subdirectory
    in the original dataset.

    Args:
        dataset_name (str): The path to the original dataset directory.
        percentage (float): The percentage of images to include in the subset (between 0 and 100).
        new_path (str): The path to the directory where the subset dataset will be created.

    Returns:
        None

    """
    print(f"Creating {percentage}% data subset for the 'train' subdirectory...")
    subset_percentage = percentage / 100

    # Set the paths for the original and new directories
    dataset_path = Path(dataset_name)
    new_directory = Path(new_path)

    # Keep the "test" and "val" directories unchanged
    tvt_dirs = [
        subdir
        for subdir in dataset_path.iterdir()
        if subdir.is_dir() and subdir.name != "train"
    ]

    for tvt in tvt_dirs:
        tvt_path = new_directory / tvt.name
        tvt_path.mkdir(parents=True, exist_ok=True)

        # Copy the subdirectories as they are
        for class_path in tvt.iterdir():
            class_name = class_path.name
            new_class_path = tvt_path / class_name
            new_class_path.mkdir(parents=True, exist_ok=True)
            for image_path in class_path.iterdir():
                new_image_path = new_class_path / image_path.name
                copyfile(image_path, new_image_path)

    # Create a subset for the "train" subdirectory
    train_path = dataset_path / "train"
    train_new_path = new_directory / "train"
    train_new_path.mkdir(parents=True, exist_ok=True)

    class_stats = {}
    for class_path in train_path.iterdir():
        class_name = class_path.name
        class_stats[class_name] = list(class_path.glob("*"))

    # Calculate the size of the new dataset based on the subset percentage
    total_count = sum(len(images) for images in class_stats.values())
    new_dataset_size = int(total_count * subset_percentage)

    new_dataset = []
    print(f"train: {total_count}/{new_dataset_size} images selected.")

    for class_name in class_stats.keys():
        images = class_stats[class_name]
        num_images = len(images)

        num_selected_images = int(new_dataset_size / len(class_stats))
        # Randomly select the images from the current class
        selected_images = random.sample(images, num_selected_images)
        # Add the selected images to the new dataset
        new_dataset.extend(selected_images)

        # Copy the selected images to the new directory
        for image_path in new_dataset:
            class_name = image_path.parent.name
            new_image_path = train_new_path / class_name / image_path.name
            new_image_path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(image_path, new_image_path)

    # Calculate and print statistics
    print("Selected images per class:")
    for class_name, images in class_stats.items():
        selected_images = [image for image in images if image in new_dataset]
        selected_count = len(selected_images)
        total_count = len(images)
        print(
            f"  > Class: {class_name}, Selected: {selected_count}/{total_count} images"
        )

    print("Subset dataset created successfully!\n")


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


# View an image
def view_random_image(root_path):
    """
    root_path: root folder path
    """
    # Get all image file paths in the directory
    image_paths = list(Path(root_path).rglob("*.jpeg"))

    if len(image_paths) == 0:
        print("No image files found in the specified directory.")
        return

    # Select a random image path
    random_image_path = random.choice(image_paths)
    # Read in the image and plot it using matplotlib
    img = mpimg.imread(random_image_path)
    plt.imshow(img)
    plt.title(str(random_image_path.parent.name))
    plt.axis("off")

    # Get the image size
    height, width, channels = img.shape

    # Print the size statistics
    print("Image Path:", random_image_path)
    print("Width:", width)
    print("Height:", height)
    print("Channels:", channels)


# Plot loss curves of a model with matplotlib
def plot_loss_curves_mplt(history):
    """Plots training curves of a results dictionary.

    Args:
        history (dict): dictionary containing training history, e.g.
            {"loss": [...],
             "val_loss": [...],
             "accuracy": [...],
             "val_accuracy": [...]}
    """
    loss = history.history["loss"]
    test_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    test_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Plot loss curves of a model using plotly.py
def plot_loss_curves_plotly(history):
    """Plots training curves of a results dictionary.

    Args:
        history (dict): dictionary containing training history, e.g.
            {"loss": [...],
             "val_loss": [...],
             "accuracy": [...],
             "val_accuracy": [...]}
    """
    loss = history.history["loss"]
    test_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    test_accuracy = history.history["val_accuracy"]
    epochs = list(range(len(loss)))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

    # Plot loss
    fig.add_trace(
        go.Scatter(
            x=epochs, y=loss, mode="lines", name="train_loss", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=test_loss, mode="lines", name="val_loss", line=dict(color="red")
        ),
        row=1,
        col=1,
    )

    # Plot accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=accuracy,
            mode="lines",
            name="train_accuracy",
            line=dict(color="blue"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=test_accuracy,
            mode="lines",
            name="val_accuracy",
            line=dict(color="red"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=500, width=1000, title_text="Training Curves")

    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=2)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    fig.show()


# Create tensorboard callback (functionized because need to create a new one for each model)


def create_tensorboard_callback(dir_name, experiment_name, timezone="Asia/Dhaka"):
    """
    Creates a TensorBoard callback for logging training metrics during model training.

    Parameters:
    dir_name (str): Directory name where the TensorBoard logs will be saved.
    experiment_name (str): Name of the experiment or model for organizing the logs.
    timezone (str): Timezone for log directory naming. Default is "Asia/Dhaka".

    Returns:
    tf.keras.callbacks.TensorBoard: TensorBoard callback for logging training metrics.
    """
    # Set the time zone based on the provided argument
    target_time_zone = pytz.timezone(timezone)

    # Get the current time in the system's local time zone
    current_time = datetime.datetime.now()

    # Convert the current time to the target time zone
    target_current_time = current_time.astimezone(target_time_zone)

    # Format the current time for the log directory name
    log_time = target_current_time.strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = f"{dir_name}/{experiment_name}/{log_time}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def create_feature_extractor_model(model_url, IMAGE_SHAPE, num_classes):
    """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
      model_url (str): A TensorFlow Hub feature extraction URL.
      num_classes (int): Number of output neurons in output layer,
        should be equal to number of target classes, default 10.

    Returns:
      An uncompiled Keras Sequential model with model_url as feature
      extractor layer and Dense output layer with num_classes outputs.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(
        model_url,
        trainable=False,  # freeze the underlying patterns
        name="feature_extraction_layer",
        input_shape=IMAGE_SHAPE + (3,),
    )  # define the input image shape

    # Create our own model
    model = tf.keras.Sequential(
        [
            feature_extractor_layer,  # use the feature extraction layer as the base
            tf.keras.layers.Dense(
                num_classes, activation="softmax", name="output_layer"
            ),  # create our own output layer
        ]
    )

    return model


def augment_random_image(target_dir, data_augmentation):
    """
    Reads a random image from a randomly chosen class in the target directory,
    applies data augmentation to the image, and plots the original and augmented images side by side.

    Parameters:
    target_dir (str): Path to the directory containing the target images.
    data_augmentation (tf.keras.Sequential): Data augmentation model to apply to the images.

    Returns:
    None
    """
    # Get a random class from the target directory
    target_class = random.choice(os.listdir(target_dir))

    # Get a random image from the chosen class
    class_dir = os.path.join(target_dir, target_class)
    random_image = random.choice(os.listdir(class_dir))
    random_image_path = os.path.join(class_dir, random_image)

    # Read the random image
    img = mpimg.imread(random_image_path)

    # Augment the image
    augmented_img = data_augmentation(tf.expand_dims(img, axis=0))

    # Plot the original and augmented images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original random image from class: {target_class}")
    plt.axis(False)

    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(augmented_img) / 255.0)
    plt.title(f"Augmented random image from class: {target_class}")
    plt.axis(False)
    plt.show()


def compare_histories_plotly(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects and plots the training and validation accuracy and loss using Plotly.

    Parameters:
    original_history (keras.callbacks.History): History object of the original model.
    new_history (keras.callbacks.History): History object of the new model.
    initial_epochs (int): Number of initial epochs before fine-tuning. Default is 5.

    Returns:
    None
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Training and Validation Accuracy",
            "Training and Validation Loss",
        ),
    )

    # Add traces to the first subplot (Training Accuracy and Validation Accuracy)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_acc))),
            y=total_acc,
            mode="lines",
            name="Training Accuracy",
            marker=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_val_acc))),
            y=total_val_acc,
            mode="lines",
            name="Validation Accuracy",
            marker=dict(color="red"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[initial_epochs - 1, initial_epochs - 1],
            y=[min(total_acc), max(total_acc)],
            mode="lines",
            name="Fine-tune starting point",
            marker=dict(color="black"),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)

    # Add traces to the second subplot (Training Loss and Validation Loss)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_loss))),
            y=total_loss,
            mode="lines",
            name="Training Loss",
            marker=dict(color="blue"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_val_loss))),
            y=total_val_loss,
            mode="lines",
            name="Validation Loss",
            marker=dict(color="red"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[initial_epochs - 1, initial_epochs - 1],
            y=[min(total_loss), max(total_loss)],
            mode="lines",
            name="Fine-tune starting point",
            marker=dict(color="black"),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    # Update layout and display the figure
    fig.update_layout(
        height=1000,
        width=1000,
        showlegend=True,
        title="Comparison of Training and Validation Metrics",
    )
    fig.show()


def compare_histories_mplt(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_val_acc, label="Validation Accuracy")
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
    )  # reshift plot around epochs
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
    )  # reshift plot around epochs
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()
