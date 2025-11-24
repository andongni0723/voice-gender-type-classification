from pathlib import Path

import torch.cuda
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.early_stopping import EarlyStopper
from src.model import VoiceClassifyModel


def device_detected() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def train_with_dataloader(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | None = None
) -> VoiceClassifyModel:
    device = device or torch.device(device_detected())
    cnn_model = VoiceClassifyModel(classify_class=2).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    early_stopper = EarlyStopper(patience=8, mode='max', delta=1e-4)
    total_step = 0
    for epoch in range(epochs):

        cnn_model.train()
        running_loss: float = 0.0
        running_correct: int = 0
        running_total: int = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x: Tensor = batch_x.to(device)  # (B, 1, H, W)
            batch_y: Tensor = batch_y.to(device)  # (B)
            # if epoch == 1: draw.draw_batch_imgs('train/imgs', batch_x, step=batch_idx)
            optimizer.zero_grad()
            logits = cnn_model(batch_x)  # (B, num_class)
            loss = loss_func(logits, batch_y)
            loss.backward()
            optimizer.step()
            # draw.draw_line('train/loss', loss.item(), step=total_step)

            # metrics
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            correct = (preds == batch_y).sum().item()
            running_total += batch_y.size(dim=0)
            running_correct += correct
            running_loss += loss.item() * batch_y.size(dim=0)
            if batch_idx % (len(train_loader) // 2) == 0:
                batch_acc = correct / batch_y.size(0)
                print(f'epoch {epoch} batch {batch_idx} loss {loss.item():.6f} batch_acc {batch_acc:.3f}')
            total_step += 1

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        # draw.draw_line('train/epoch_acc', epoch_acc, epoch)
        # draw.draw_line('train/epoch_loss', epoch_loss, step=epoch)
        print(f'==> Epoch {epoch} Train loss: {epoch_loss:.6f}, Train acc: {(epoch_acc * 100):.4f}%')

        # Valid
        cnn_model.eval()
        val_loss: float = 0.0
        val_correct: int = 0
        val_total: int = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx: Tensor = bx.to(device)
                by: Tensor = by.to(device)
                out = cnn_model(bx)
                l = loss_func(out, by)
                preds = torch.argmax(out, dim=1)
                val_loss += l.item() * by.size(dim=0)
                val_correct += (preds == by).sum().item()
                val_total += by.size(dim=0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        # draw.draw_line('valid/epoch_acc', val_acc, epoch)
        # draw.draw_line('valid/epoch_loss', val_loss, step=epoch)
        print(f'==> Epoch {epoch} Val loss: {val_loss:.6f}, Val acc: {(val_acc * 100):.4f}%')

        # Early Stopping
        monitor_value = val_acc
        should_stop = early_stopper.step(monitor_value, cnn_model)
        # draw.draw_line('valid/best_acc', early_stopper.best, epoch)
        if should_stop:
            print(f'[EarlyStop] Stop training, model no improved for {early_stopper.patience} epochs.'
                  f'Restoring best model (best={early_stopper.best}) and stop.')
            early_stopper.restore_best(cnn_model)
            break

    print('Train done!')
    return cnn_model

def save_model(out_path: Path, model: nn.Module, verbose: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    if verbose: print(f'Model saved to: {out_path}')