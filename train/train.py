import torch
import datetime
from tqdm import trange

from torch.amp import GradScaler, autocast

# cuda effeciency
scaler = GradScaler()  # No arguments needed

def train(n_epochs, optimizer, model, loss_fn, train_loader, device, val_loader=None, scheduler=None):
    n_train_batch = len(train_loader)
    losses_train = []
    scores_train = []

    if val_loader is not None:
        n_val_batch = len(val_loader)
        losses_val = []

    # model.train()
    # optimizer.zero_grad()

    model = model.to(device)
    
    for epoch in trange(1, n_epochs + 1):
        loss_train = 0.0
        loss_val = 0.0

        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()   # new

            ### commented out to text cuda stuff
            # outputs = model(imgs)
            # loss = loss_fn(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

             # Mixed precision forward pass
            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Unscale before clearing memory
            scaler.unscale_(optimizer)  # 🔹 Added this line before clearing cache
            torch.cuda.empty_cache()    

            scaler.step(optimizer)
            scaler.update()
            loss_train += loss.item()

            del imgs, labels, outputs, loss  # Clear memory, new

        losses_train.append(loss_train / n_train_batch)

        if val_loader is not None:
            model.eval()
            with torch.inference_mode():    # changed to torch.inference_mode() from torch.no_grad()
                for imgs, labels in val_loader:
                    # model.eval()
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)
                    loss_val += loss.item()

                    del imgs, labels, outputs, loss  # Clear memory, new
                    torch.cuda.empty_cache()   # Clear cache

                losses_val.append(loss_val / n_val_batch)

        #   if epoch == 1 or epoch % 5 == 0:
        # print for every epoch
        print(f'--------- Epoch: {epoch} ---------')
        print('Training loss {:.5f} at {}'.format(loss_train / n_train_batch, datetime.datetime.now()))
        if val_loader is not None:
            print('Validation loss {:.5f} at {}'.format(loss_val / n_val_batch, datetime.datetime.now()))
        print()
        if scheduler:
            scheduler.step()
            print(f'Learning rate updated: {optimizer.param_groups[0]["lr"]:.6f}')
        print()

    return losses_train#, losses_val