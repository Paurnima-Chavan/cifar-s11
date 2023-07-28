
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    plt.figure(dpi=500);

    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.clip(np.transpose(img, (1, 2, 0)), 0, 1), cmap='viridis', interpolation='bilinear')



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
    

# Function to plot misclassified images
def plot_misclassified_images(images, pred_labels, correct_labels):

    plt.figure(dpi=500);
    fig, axes = plt.subplots(4, 5, figsize=(15, 7))
    fig.suptitle("Misclassified Images", fontsize=8)

    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().permute(1, 2, 0)
        ax.imshow(np.clip(img, 0, 1),  cmap='viridis', interpolation='bilinear')
        ax.axis("off")
        ax.set_title(f"Pred: {pred_labels[i]}, Target: {correct_labels[i]}")

    plt.tight_layout()
    plt.show()
