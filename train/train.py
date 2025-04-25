import torch
import datetime
from tqdm import trange
from torch.amp import GradScaler, autocast

def train(model, optimizer, loss_fn, train_loader,val_loader=None, device="cuda", n_epochs=30, scheduler=None, early_stopping=False, patience=5):
    n_train_batch = len(train_loader)
    losses_train = []
    model = model.to(device)

    if val_loader is not None:
        n_val_batch = len(val_loader)
        losses_val = []

    # cuda effecient
    scaler = GradScaler()
    
    progress_bar = trange(1, n_epochs + 1)
    for epoch in progress_bar:
        loss_train = 0.0
        loss_val = 0.0
        best_epoch_val_loss = float('-inf')
        epochs_no_improve = 0

        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            ### commented out to if not using autocast 
            # # Normal training
            # outputs = model(imgs)
            # loss = loss_fn(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            # Autocast
            optimizer.zero_grad()  
            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.cuda.empty_cache()    
            scaler.step(optimizer)
            scaler.update()

            loss_train += loss.item()
            del imgs, labels, outputs, loss  # Clear memory

        losses_train.append(loss_train / n_train_batch)
    
        if val_loader is not None:
            model.eval()
            with torch.inference_mode():    # if not using autocast (?): torch.no_grad()
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)
                    loss_val += loss.item()

                    del imgs, labels, outputs, loss  # Clear memory
                    torch.cuda.empty_cache()   # Clear cache

                losses_val.append(loss_val / n_val_batch)

        # if epoch == 1 or epoch % 5 == 0:
        # print(f'--------- Epoch: {epoch} ---------')
        # print('Training loss {:.5f} at {}'.format(loss_train / n_train_batch, datetime.datetime.now()))
        # if val_loader is not None:
        #     print('Validation loss {:.5f} at {}'.format(loss_val / n_val_batch, datetime.datetime.now()))
        # print()

        if scheduler:
            scheduler.step()

        if early_stopping and val_loader is not None:
            if losses_val[-1] > best_epoch_val_loss:
                best_epoch_val_loss = losses_val[-1]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} with patience {patience}")
                    break
        
        progress_bar.set_postfix({
            'Epoch': epoch,
            'Learning Rate': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'Training Loss': f'{loss_train / n_train_batch:.5f}',
            'Validation Loss': f'{loss_val / n_val_batch:.5f}' if val_loader is not None else 'N/A'
        })

    return losses_train, losses_val