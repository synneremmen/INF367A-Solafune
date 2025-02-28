import torch
import datetime

def train(n_epochs, optimizer, model, loss_fn, train_loader, device, val_loader=None):

    n_train_batch = len(train_loader)
    losses_train = []
    scores_train = []
    if val_loader is not None:
        n_val_batch = len(val_loader)
        losses_val = []

    model.train()
    optimizer.zero_grad()
    model = model.to(device)
    
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        loss_val = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            model.train()

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()
        losses_train.append(loss_train / n_train_batch)

        if val_loader is not None:
            with torch.no_grad():
                for imgs, labels in val_loader:
                    model.eval()

                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)
                    loss_val += loss.item()

                losses_val.append(loss_val / n_val_batch)

        if epoch == 1 or epoch % 5 == 0:
            print(f'--------- Epoch: {epoch} ---------')
            print('Training loss {:.5f} at {}'.format(loss_train / n_train_batch, datetime.datetime.now()))
            if val_loader is not None:
                print('Validation loss {:.5f} at {}'.format(loss_val / n_val_batch, datetime.datetime.now()))
            print()

    return losses_train#, losses_val