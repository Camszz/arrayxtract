import torch
import sys
import math
from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()

    print(f"--- Epoch: {epoch} ---")

    total_loss = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    batch_idx = 0
    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)
    
        optimizer.zero_grad()

        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
                lr_scheduler.step()

        # Accumulate individual losses
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()

        # Sum the losses
        total_loss += losses.item()

        batch_idx += 1

    average_loss = total_loss / len(data_loader)
    average_loss_classifier = total_loss_classifier / len(data_loader)
    average_loss_box_reg = total_loss_box_reg / len(data_loader)

    print(f"Average train losses --- total : {average_loss:.4f} --- loss_classifier: {average_loss_classifier:.4f} --- loss_box_reg: {average_loss_box_reg:.4f}")

    return average_loss, average_loss_classifier, average_loss_box_reg

def train(model, optimizer, train_loader, val_loader, device, num_epochs, lr_scheduler, print_freq = 10, scaler=None):
    train_losses = {}
    val_losses = {}
    for epoch in range(num_epochs):
        train_losses[epoch] = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq)
        lr_scheduler.step()
        val_losses[epoch] = validate(model, val_loader, device)

    return train_losses, val_losses


def validate(model, data_loader, device):
    total_loss = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)  # Move images to the appropriate device (CPU/GPU)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]  # Move targets to the appropriate device

        # Forward pass
        with torch.no_grad():
            loss_dict = model(images, targets)
        
        # Accumulate individual losses
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()

        # Sum the losses
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        num_batches += 1

    average_loss = total_loss / num_batches
    average_loss_classifier = total_loss_classifier / num_batches
    average_loss_box_reg = total_loss_box_reg / num_batches

    print(f'Average validation losses --- total : {average_loss:.4f} --- loss_classifier: {average_loss_classifier:.4f} --- loss_box_reg: {average_loss_box_reg:.4f}')

    return average_loss, average_loss_classifier, average_loss_box_reg