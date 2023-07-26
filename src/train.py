from tqdm import tqdm

train_losses = []
train_acc = []
lrs = []


def get_lr(optimizer):
    """"
        for tracking how learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, device, train_loader, optimizer, epoch, scheduler, criterion):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do back-propagation because PyTorch
        # accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that
        # you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)  # F.nll_loss(y_pred, target)
        train_losses.append(loss)
        lrs.append(get_lr(optimizer))

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()}  LR = {get_lr(optimizer)} '
                                  f'Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        train_acc.append(100 * correct / processed)


def reset_train_loss():
    global train_losses, train_acc, lrs
    train_losses = []
    train_acc = []
    lrs = []


def get_train_loss_acc():
    return train_losses, train_acc
