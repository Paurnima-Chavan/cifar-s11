import torch
from gradcam import GradCAM
import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def show_misclassified_images_using_grad_cam(model, misclassified_images, pred_labels, correct_labels, no_images):
    gradcam = GradCAM(model, target_layer=model.layer4[-1])

    target_class = 3

    plt.figure(dpi=500);
    fig, axs = plt.subplots(4, 5, figsize=(15, 7));
    fig.suptitle("Misclassified Images", fontsize=8)

    for i in range(no_images):
        image = misclassified_images[i]

        image1 = image.reshape([1, 3, 32, 32])

        # Generate Grad-CAM for the misclassified image
        heatmap, _ = gradcam(image1, target_class)

        npimg = denormalize_image(image)

        heatmap = heatmap.squeeze().cpu().numpy()
        cam_image = show_cam_on_image(npimg, heatmap, use_rgb=True, image_weight=0.7)

        # Display toverlayhe original image and Grad-CAM overlay
        axs[i // 5, i % 5].imshow(cam_image, cmap='viridis', interpolation='bilinear')
        axs[i // 5, i % 5].set_title(f"Pred: {pred_labels[i]}, Target: {correct_labels[i]}")
        axs[i // 5, i % 5].axis('off')

    plt.tight_layout()
    plt.show()
    pass


def denormalize_image(image):
    # Denormalize and convert to numpy for visualization
    npimg = image.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = ((npimg * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
    return npimg


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
