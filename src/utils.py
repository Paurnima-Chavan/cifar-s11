

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))



def plot_dataset_images(train_loader, no_images):
    """
    This will plot 'n' (no_images) images for given dataset
    :param train_loader: dataset
    :param no_images: number of images to plot
    :return:
    """
    import math

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # convert images to numpy for display

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(no_images):
        ax = fig.add_subplot(2, math.ceil(no_images / 2), idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])


def plot_train_test_accuracy_loss(train_losses, train_acc, test_losses, test_acc):
    """
    This function is used to plot the training and testing accuracy as well as the training and testing loss.
    It creates a 2x2 grid of subplots in a figure to visualize the four plots.
    :return:
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")