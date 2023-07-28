import torch
from gradcam import GradCAM
import matplotlib.pyplot as plt
import numpy as np


def get_misclassified_images(model, test_loader, device, no_images=20):
    misclassified_images = []
    misclassified_labels = []
    correct_labels = []
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    # model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            misclassified_idx = (pred != target).nonzero(as_tuple=True)[0]
            misclassified_images.extend(data[misclassified_idx])
            misclassified_labels.extend(pred[misclassified_idx])
            correct_labels.extend(target[misclassified_idx])

            # Map label indices to class names
            misclassified_labels = [classes[label] for label in misclassified_labels]
            correct_labels = [classes[label] for label in correct_labels]

            if len(misclassified_images) >= no_images:
                break

    return misclassified_images, misclassified_labels, correct_labels


def show_misclassified_images_using_grad_cam(model, misclassified_images, pred_labels, correct_labels, no_images):

    gradcam = GradCAM(model, target_layer=model.layer4[-1])

    target_class = 3

    fig, axs =  plt.subplots(4, 5, figsize=(15, 7))
    fig.suptitle("Misclassified Images", fontsize=8)

    for i in range(no_images):
        image = misclassified_images[i]

        image1 = image.reshape([1, 3, 32, 32])

        # Generate Grad-CAM for the misclassified image
        heatmap, _ = gradcam(image1, target_class)

        # Denormalize and convert to numpy for visualization
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        heatmap = heatmap.squeeze().cpu().numpy()

        # Superimpose the heatmap on the image
        alpha = 0.5
        overlay = image * (1 - alpha) + np.dstack([heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap)]) * alpha

        # Display toverlayhe original image and Grad-CAM overlay
        c = axs[i // 5, i % 5].imshow(np.clip(overlay, 0, 1))
        a = axs[i // 5, i % 5].set_title(f"Pred: {pred_labels[i]}, Target: {correct_labels[i]}")
        b = axs[i // 5, i % 5].axis('off')

    plt.tight_layout()
    plt.show()

