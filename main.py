import torch
import torch.optim as optim
from utils import plot_dataset_images, plot_train_test_accuracy_loss, plot_misclassified_images
from pytorch_grad_cam import get_misclassified_images, show_misclassified_images_using_grad_cam
from dataset import load_cifar10_data
from models.resnet import ResNet18, model_summary

from train import train, reset_train_loss, get_train_loss_acc
from test import test, reset_test_loss, get_test_loss_acc
from torch_lr_finder import LRFinder
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

def classify_images_for_cifar10():
    batch_size = 512
    train_loader, test_loader = load_cifar10_data(batch_size=batch_size)
    plot_dataset_images(train_loader, 20)
    # Load the model
    SEED = 1
    # CUDA?
    use_cuda = torch.cuda.is_available()
    # For reproducibility
    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")

    #Load model
    model = ResNet18().to(device)
    model_summary(model, input_size=(3, 32, 32))

    # using one cycle policy find LR
    optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=120, step_mode="exp")
    print(lr_finder.best_loss)
    lr_finder.plot()  # to inspect loss-learning rate graph
    lr_finder.reset()  # to reset the model amd optimizer to their initial state

    EPOCHS = 20
    reset_train_loss()
    reset_test_loss()

    optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = OneCycleLR(
        optimizer,
        max_lr=4.61E-02,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=2 / EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear')
    
    #Model Training
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")
        train(model, device, train_loader, optimizer, epoch, scheduler, criterion)
        test(model, device, test_loader, criterion)

    # Plot Graph to show loss curves for test and train datasets
    train_losses, train_acc = get_train_loss_acc()
    test_losses, test_acc = get_test_loss_acc()

    train_losses = torch.tensor(train_losses, device='cpu')
    train_acc = torch.tensor(train_acc, device='cpu')
    test_losses = torch.tensor(test_losses, device='cpu')
    test_acc = torch.tensor(test_acc, device='cpu')

    plot_train_test_accuracy_loss(train_losses, train_acc, test_losses, test_acc)

    #how gradcamLinks to an external site. output on 10 misclassified images. 
    misclassified_images, pred_labels, correct_labels = get_misclassified_images(model, test_loader, device)
    show_misclassified_images_using_grad_cam(model, misclassified_images, pred_labels, correct_labels, 20)

    #show a gallery of 10 misclassified images
    plot_misclassified_images(misclassified_images, pred_labels, correct_labels)



